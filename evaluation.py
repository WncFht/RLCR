import copy
import gc
import json
import math
import os
import re

import datasets
import numpy as np
import torch
from dataset_processing import process_dataset
from datasets import load_dataset
from eval.check_functions import confidence_verifier, llm_confidence_verifier
from eval.eval_args import GlobalArgs, LocalConfig
from eval.eval_utils import hash_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams


def main(global_args, local_configs, debug_mode=False):
    # 🌟 DEBUG MODE 逻辑 (1/2): 限制样本大小
    if debug_mode and global_args.sample_size is None:
        global_args.sample_size = 5
        print(f"DEBUG MODE activated: Setting sample_size to {global_args.sample_size}")
    elif debug_mode:
        print(
            f"DEBUG MODE activated: Using existing sample_size = {global_args.sample_size}"
        )

    metrics_path = (
        os.path.join(global_args.log_path, "metrics.json")
        if global_args.log_path is not None
        else None
    )
    existing_metrics = {}
    if metrics_path and os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                existing_metrics = json.load(f)
            print(
                f"Loaded existing metrics for {len(existing_metrics)} configs from {metrics_path}"
            )
        except Exception as exc:
            print(f"Failed to load existing metrics from {metrics_path}: {exc}")
            existing_metrics = {}

    try:
        dataset = datasets.load_from_disk("./" + global_args.dataset_name)
    except:
        dataset = load_dataset(global_args.dataset_name)
    dataset = dataset[global_args.split]
    dataset = dataset.map(lambda x: hash_dataset(x, global_args.hash_key))

    # 应用样本大小限制
    if global_args.sample_size is not None:
        dataset = dataset.select(range(global_args.sample_size))

    final_dataset = copy.deepcopy(dataset)

    all_metrics = {}
    updated = False
    run_metrics = {}
    try:
        existing_dataset = load_dataset(global_args.store_name, split=global_args.split)
        print(
            f"Found existing dataset {global_args.store_name} with {len(existing_dataset)} samples"
        )
        final_dataset = copy.deepcopy(existing_dataset)
        # ⚠️ 注意: 在 Debug 模式下，如果加载了现有的 dataset，你需要确保它也被限制在前 N 个样本
        if debug_mode and global_args.sample_size is not None:
            final_dataset = final_dataset.select(range(global_args.sample_size))
    except:
        try:
            existing_dataset = datasets.load_from_disk(global_args.store_name)
            print(
                f"Found existing dataset {global_args.store_name} with {len(existing_dataset)} samples"
            )
            final_dataset = copy.deepcopy(existing_dataset)
            # ⚠️ 注意: 在 Debug 模式下，如果加载了现有的 dataset，你需要确保它也被限制在前 N 个样本
            if debug_mode and global_args.sample_size is not None:
                final_dataset = final_dataset.select(range(global_args.sample_size))
        except:
            print(f"No existing dataset found for {global_args.store_name}")
            existing_dataset = None

    for config in local_configs:
        config.split = global_args.split
        if global_args.fresh:
            config.fresh = True
        out_dict = None
        available = False
        run_metrics[config.name] = {}

        if config.name in existing_metrics:
            print(
                f"Skipping {config.name} because metrics already exist in metrics.json"
            )
            continue

        if existing_dataset is not None:
            if f"{config.name}-output_0" in existing_dataset.column_names:
                available = True
                if not config.fresh:
                    print(f"Skipping {config.name} because it already exists")
                    continue
                else:
                    updated = True
                    print(f"Overwriting {config.name} because fresh is True")
        name = config.name
        config.dataset_name = global_args.dataset_name
        local_dataset = copy.deepcopy(dataset)
        local_dataset = process_dataset(local_dataset, config)

        ##### GENERATION #####

        # 仅在非 Debug 模式下加载LLM，或者在必要时加载 (例如 for 'gen_then_classify' 模式)
        # 为简洁起见，保留了原有的LLM加载逻辑，假设你需要LLM来生成前5个样本的输出

        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        to_tokenize = [
            local_dataset[i][config.tokenize_key] for i in range(len(local_dataset))
        ]
        prompt_ids = tokenizer.apply_chat_template(
            to_tokenize, add_generation_prompt=True
        )
        texts = [tokenizer.decode(x) for x in prompt_ids]

        print("Prompt samples for config: ", name)
        print(tokenizer.decode(prompt_ids[0]))

        sampling_params = SamplingParams(
            n=config.n,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
            logprobs=1,
        )
        llm = LLM(
            model=config.model,
            gpu_memory_utilization=global_args.gpu_memory_utilization,
        )
        outputs = llm.generate(texts, sampling_params=sampling_params)

        ##### POST-GENERATION PROCESSING #####

        if "ans_at_end" in config.vllm_task:
            inst = "Thinking time ended \n\n. My final answer is "
            prompts = []
            for text, output in zip(texts, outputs):
                for i in range(config.n):
                    prompts.append(text + output.outputs[i].text + inst)
            ans_sampling_params = SamplingParams(n=1, temperature=0, max_tokens=50)
            ans_outputs = llm.generate(prompts, sampling_params=ans_sampling_params)

            ans_calls_needed = 0
            counter = 0
            for out in outputs:
                for j in range(config.n):
                    # first try to extract the answer from the output
                    ans_pattern = r"<answer>(.*?)</answer>"
                    ans_matches = re.findall(
                        ans_pattern, out.outputs[j].text, re.DOTALL | re.MULTILINE
                    )  # Get all <answer>...</answer> occurrences
                    last_answer = (
                        ans_matches[-1] if ans_matches else ""
                    )  # Get the last answer, if exists
                    ## ONLY IF NO ANSWER IS FOUND, USE THE ANSWER FROM THE ANS_OUTPUTS
                    if last_answer == "":
                        last_answer = ans_outputs[counter].outputs[0].text
                        out.outputs[j].text = (
                            out.outputs[j].text
                            + "<answer> "
                            + last_answer
                            + " </answer>"
                        )
                        ans_calls_needed += 1
                    counter += 1
            print(
                f"Number of answer calls needed for {config.name}: {ans_calls_needed / (config.n * len(outputs))}"
            )
            run_metrics[config.name]["ans_calls_needed"] = ans_calls_needed / (
                config.n * len(outputs)
            )

        if "gen_then_classify" in config.vllm_task:
            del llm
            gc.collect()

            ques_key = (
                "problem" if "problem" in local_dataset.column_names else "question"
            )

            if config.split_at_confidence:
                print(f"Splitting at confidence for {config.name}")
                for output in outputs:
                    # keep part before <confidence>
                    output.outputs[0].text = output.outputs[0].text.split(
                        "<confidence>"
                    )[0]

            # now append the generated text to the original prompt
            texts = [
                f"\n\nPROBLEM: {local_dataset[i][ques_key]}\n\nEND OF PROBLEM\n\nMODEL'S RESPONSE: {output.outputs[0].text}\n\nEND OF RESPONSE\n\n"
                for i, output in enumerate(outputs)
            ]
            print("Gen and Classify Samples for config: ", name)
            print(texts[0])
            print(texts[1])
            class_outputs = []
            if config.use_hf:
                llm = AutoModelForSequenceClassification.from_pretrained(
                    config.class_model
                ).to("cuda")
                # set to eval mode
                llm.eval()
                tokenizer = AutoTokenizer.from_pretrained(config.class_model)
                class_outputs = []
                batch_size = 16
                for i in tqdm(
                    range(0, len(texts), batch_size), desc="Classifying texts"
                ):
                    batch_texts = (
                        texts[i : i + batch_size]
                        if i + batch_size <= len(texts)
                        else texts[i:]
                    )
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=4096,
                    )
                    with torch.no_grad():
                        inputs = inputs.to("cuda")
                        output = llm(**inputs)
                        output_tensor = output.logits.cpu().detach().numpy()
                        float_probs = output_tensor[
                            :, 0
                        ]  # Get first column for each item in batch
                        # convert to probabilities using sigmoid
                        fps = []
                        for fp in float_probs:
                            fps.append(1 / (1 + math.exp(-fp)))
                        class_outputs.extend(fps)
            else:
                llm = LLM(
                    model=config.class_model,
                    task="classify",
                    gpu_memory_utilization=global_args.gpu_memory_utilization,
                )
                class_outputs = llm.classify(texts)

            del llm
            gc.collect()

        if "confidence_prob" in config.vllm_task:
            token = tokenizer.tokenize("answer")
            invalid_count = 0
            # Get the logprob for everything inside <answer> </answer>
            for output in outputs:
                for i in range(config.n):
                    picked = output.outputs[i]
                    len_gen = len(picked.logprobs)
                    tokens = []
                    probs = []
                    for j in range(len_gen):
                        lp_val = next(iter(picked.logprobs[j].values())).logprob
                        token = next(iter(picked.logprobs[j].values())).decoded_token
                        probs.append(np.exp(lp_val))
                        tokens.append(token)
                    # find the 2nd last and last occurence of token
                    answer_indices = [
                        i for i, token in enumerate(tokens) if token == "answer"
                    ]

                    # Get last and second last, if available
                    end_index = answer_indices[-1] if len(answer_indices) >= 1 else None
                    start_index = (
                        answer_indices[-2] if len(answer_indices) >= 2 else None
                    )
                    if (
                        start_index == None
                        or end_index == None
                        or end_index - start_index >= 30
                    ):
                        output.outputs[i].text = (
                            output.outputs[i].text + "<confidence> 0.5 </confidence>"
                        )
                        invalid_count += 1
                    else:
                        selected_probs = probs[start_index:end_index]
                        selected_tokens = tokens[start_index:end_index]
                        avg_prob = sum(selected_probs) / len(selected_probs)
                        output.outputs[i].text = (
                            output.outputs[i].text
                            + f"<confidence> {avg_prob} </confidence>"
                        )

            print(
                f"Number of invalid confidence calls for {config.name}: {invalid_count / (config.n * len(outputs))}"
            )
            run_metrics[config.name]["invalid_confidence_prob_calls"] = (
                invalid_count / (config.n * len(outputs))
            )

        if "confidence_at_end" in config.vllm_task:
            inst = "Thinking time ended \n\n. My verbalized confidence in my answer as a number between 0 and 100 is equal to "
            prompts = []
            for text, output in zip(texts, outputs):
                for i in range(config.n):
                    prompts.append(text + output.outputs[i].text + inst)

            verb_sampling_params = SamplingParams(n=1, temperature=0, max_tokens=20)
            verb_outputs = llm.generate(prompts, sampling_params=verb_sampling_params)

            conf_calls_needed = 0
            counter = 0
            for output in outputs:
                for i in range(config.n):
                    conf_pattern = r"<confidence>(.*?)</confidence>"
                    conf_matches = re.findall(
                        conf_pattern, output.outputs[i].text, re.DOTALL | re.MULTILINE
                    )
                    last_confidence = conf_matches[-1] if conf_matches else ""
                    ## ONLY IF NO CONFIDENCE IS FOUND, USE THE CONFIDENCE FROM THE VERB_OUTPUTS
                    if last_confidence == "":
                        last_confidence = verb_outputs[counter].outputs[0].text
                        output.outputs[i].text = (
                            output.outputs[i].text
                            + "<confidence>"
                            + last_confidence
                            + "</confidence>"
                        )
                        conf_calls_needed += 1
                    counter += 1
            print(
                f"Number of confidence calls needed for {config.name}: {conf_calls_needed / (config.n * len(outputs))}"
            )
            run_metrics[config.name]["conf_calls_needed"] = conf_calls_needed / (
                config.n * len(outputs)
            )
        if out_dict is None:
            out_dict = {}

            for i in range(config.n):
                out_dict[f"{name}-output_{i}"] = []

            for output in outputs:
                for i in range(len(output.outputs)):
                    out_dict[f"{name}-output_{i}"].append(output.outputs[i].text)

        if "gen_then_classify" in config.vllm_task:
            out_dict[f"{name}-class_output"] = []
            for output in class_outputs:
                if config.use_hf:
                    out_dict[f"{name}-class_output"].append(output)
                else:
                    out_dict[f"{name}-class_output"].append(output.outputs.probs)

        for k, v in out_dict.items():
            if available:
                final_dataset = final_dataset.remove_columns([k])
            final_dataset = final_dataset.add_column(k, v)
            local_dataset = local_dataset.add_column(k, v)

        try:
            # del the llm
            del llm
            gc.collect()
        except:
            pass

        ##### CHECK FUNCTION #####
        # 1. confidence_verifier uses symbolic parsing such as exact match, math-verify (hugging face)
        # 2. llm_confidence_verifier uses a LLM to check the answer.

        if config.check_fn is not None:
            check_fn = config.check_fn
            if check_fn == "confidence_verifier":
                label_dict, metrics = confidence_verifier(
                    local_dataset, config, **config.check_fn_args
                )
            elif check_fn == "llm_confidence_verifier":
                label_dict, metrics = llm_confidence_verifier(
                    local_dataset, config, **config.check_fn_args
                )

            all_metrics[config.name] = metrics
            for k, v in label_dict.items():
                if available:
                    final_dataset = final_dataset.remove_columns([k])
                final_dataset = final_dataset.add_column(k, v)
                local_dataset = local_dataset.add_column(k, v)

    ##### END OF FOR LOOP AND CONFIG EVALUATION #####

    ##### PRINT ALL METRICS and LOG #####

    for config_name, metrics in all_metrics.items():
        print(f"Metrics for {config_name}:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    for config_name, metrics in run_metrics.items():
        print(f"Run metrics for {config_name}:")
        try:
            for k, v in metrics.items():
                print(f"{k}: {v}")
        except:
            pass
    merged_metrics = copy.deepcopy(existing_metrics)
    merged_metrics.update(all_metrics)

    if metrics_path is not None and all_metrics:
        if not os.path.exists(global_args.log_path):
            os.makedirs(global_args.log_path)
        with open(metrics_path, "w") as f:
            json.dump(merged_metrics, f, indent=4)

    # final_dataset.push_to_hub(global_args.store_name, private=True)

    # 🌟 DEBUG MODE 逻辑 (2/2): 输出 JSON
    if debug_mode:
        print("\n" + "=" * 50)
        print("🌟 DEBUG MODE RESULT (First 5 Samples) 🌟")
        print("=" * 50)
        # 使用切片确保只获取前 global_args.sample_size 个样本 (即 5 个，除非用户指定了更小的值)
        debug_output = final_dataset[: global_args.sample_size]
        print(json.dumps(debug_output, indent=4))
        print("=" * 50 + "\n")

    elif updated:
        final_dataset.save_to_disk(global_args.store_name)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="The name of the config to use")
    # ⬇️ 添加 --debug 参数
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode, limiting the dataset to the first 5 samples and printing the final output as JSON.",
    )
    args = parser.parse_args()

    # read the json file
    with open(args.config) as f:
        config = json.load(f)

    local_configs = []
    global_args = None
    for i, c in enumerate(config):
        if i == 0:
            global_args = GlobalArgs(**c)
        else:
            local_configs.append(LocalConfig(**c))

    # 传递 debug_mode 状态给 main 函数
    main(global_args, local_configs, debug_mode=args.debug)
