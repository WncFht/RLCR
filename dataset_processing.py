from system_prompts import get_sys_prompt


def process_dataset(dataset, script_args):
    sys_prompt = get_sys_prompt(script_args.sys_prompt_name)
    if script_args.sys_prompt_name in ["instruct_tabc", "instruct_tac"]:
        dataset = make_instruct_dataset(dataset, sys_prompt)
        return dataset

    if script_args.task_spec == "gen":
        dataset = make_generation_dataset(dataset, sys_prompt)

    return dataset


def make_instruct_dataset(dataset, sys_prompt):
    def make_instruct_conversation(example):
        if "question" in example.keys():
            user_format = f"{example['question']}"
        else:
            user_format = f"{example['problem']}"
        return {
            "prompt": [
                {"role": "user", "content": sys_prompt + user_format},
            ],
        }

    dataset = dataset.map(make_instruct_conversation)
    return dataset


def make_generation_dataset(dataset, sys_prompt):
    def make_generation_conversation(example):
        if "question" in example.keys():
            user_format = f"\n\nPROBLEM: {example['question']}\n\n"
        else:
            user_format = f"\n\nPROBLEM: {example['problem']}\n\n"
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ],
        }

    dataset = dataset.map(make_generation_conversation)
    return dataset


def sft_dataset_process(dataset, script_args, sys_prompt=None):
    if script_args.dataset_name == "mehuldamani/deepseek-verifier-v1":

        def mapping(example):
            user_format = f"\n\nPROBLEM: {example['problem']}\n\n"
            ans_format = f"{example['output_0']}\n{example['demo']}\n"
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
                {"role": "assistant", "content": ans_format},
            ]
            return {"messages": messages}

        dataset = dataset.map(mapping, remove_columns=["prompt"])

    return dataset


def orm_dataset_process(dataset, script_args):
    def mapping(example):
        user_format = (
            f"\n\nPROBLEM: {example['problem']}\n\n"
            f"END OF PROBLEM\n\n"
            f"MODEL'S RESPONSE: {example[script_args.orm_key]}\n\n"
            f"END OF RESPONSE\n\n"
        )
        return {"text": user_format, "label": example["label"]}

    dataset = dataset.map(mapping)

    return dataset
