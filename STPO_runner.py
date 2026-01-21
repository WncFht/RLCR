import logging
import os
import sys
from functools import partial

import datasets
import torch
import transformers
from arguments import ModelConfig, STPOConfig, STPOScriptArguments
from dataset_processing import process_dataset
from datasets import load_dataset
from reward_fns import (
    accuracy_answer_segment_reward,
    brier_confidence_segment_reward,
    brier_reward,
    confidence_one_or_zero,
    format_answer_segment_reward,
    format_confidence_segment_reward,
    format_reward,
    log_likelihood_reward,
    mean_confidence_reward,
)
from STPO_Trainer import CustomTrainer
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import TrlParser

logger = logging.getLogger(__name__)


def logger_setup(script_args, training_args, model_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")


def model_init(model_args, training_args):
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    return model_kwargs


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    logger_setup(script_args, training_args, model_args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    answer_reward_names = (
        script_args.answer_reward_funcs
        if script_args.answer_reward_funcs is not None
        else script_args.answer_selection_reward_funcs
    )
    confidence_reward_names = (
        script_args.confidence_reward_funcs
        if script_args.confidence_reward_funcs is not None
        else script_args.reward_funcs
    )

    # STPO relies on segment-level format rewards:
    # - answer selection must filter to format-correct answers (so confidence ground-truth is well-defined)
    # - confidence rollout should be constrained by a confidence-side format reward
    if not any(
        x in script_args.answer_selection_reward_funcs
        for x in ["format_answer_segment", "format"]
    ):
        raise ValueError(
            "STPO requires an answer-side format reward in answer_selection_reward_funcs "
            "(one of: 'format_answer_segment', 'format') to enable format-gated answer selection."
        )
    if not any(
        x in confidence_reward_names for x in ["format_confidence_segment", "format"]
    ):
        raise ValueError(
            "STPO requires a confidence-side format reward in confidence_reward_funcs "
            "(one of: 'format_confidence_segment', 'format') to constrain confidence rollouts."
        )

    # Confidence-training reward functions (computed on full completion)
    brier_impl = (
        brier_confidence_segment_reward
        if "format_confidence_segment" in confidence_reward_names
        else brier_reward
    )
    REWARD_FUNCS_REGISTRY = {
        "format": partial(format_reward, format_pattern=script_args.format_pattern),
        "format_answer_segment": partial(
            format_answer_segment_reward, format_pattern=script_args.format_pattern
        ),
        "format_confidence_segment": partial(
            format_confidence_segment_reward, format_pattern=script_args.format_pattern
        ),
        "accuracy": partial(
            accuracy_answer_segment_reward, format_pattern=script_args.format_pattern
        ),
        "brier": partial(brier_impl, format_pattern=script_args.format_pattern),
        "log_likelihood": partial(
            log_likelihood_reward, format_pattern=script_args.format_pattern
        ),
        "mean_confidence": mean_confidence_reward,
        "confidence_one_or_zero": confidence_one_or_zero,
    }
    confidence_reward_funcs = [
        REWARD_FUNCS_REGISTRY[func] for func in confidence_reward_names
    ]

    if script_args.confidence_reward_funcs is not None:
        # STPO uses confidence_reward_weights for training. Map it onto training_args.reward_weights
        # (used internally by the base trainer for weighting reward channels).
        if script_args.confidence_reward_weights is not None:
            training_args.reward_weights = script_args.confidence_reward_weights
        elif getattr(training_args, "reward_weights", None) is not None and len(
            training_args.reward_weights
        ) != len(confidence_reward_funcs):
            raise ValueError(
                "confidence_reward_funcs is set but training_args.reward_weights length does not match; "
                "please set confidence_reward_weights explicitly or remove reward_weights from the config "
                f"({len(training_args.reward_weights)} vs {len(confidence_reward_funcs)})."
            )

    # Answer-selection reward functions (computed on answer-only completion)
    sel_pattern = script_args.answer_selection_format_pattern or "ta"
    SELECTION_REWARD_FUNCS_REGISTRY = {
        "format": partial(format_reward, format_pattern=sel_pattern),
        "format_answer_segment": partial(
            format_answer_segment_reward, format_pattern=sel_pattern
        ),
        "format_confidence_segment": partial(
            format_confidence_segment_reward, format_pattern=sel_pattern
        ),
        "accuracy": partial(accuracy_answer_segment_reward, format_pattern=sel_pattern),
        "brier": partial(brier_reward, format_pattern=sel_pattern),
        "log_likelihood": partial(log_likelihood_reward, format_pattern=sel_pattern),
        "mean_confidence": mean_confidence_reward,
        "confidence_one_or_zero": confidence_one_or_zero,
    }
    answer_selection_reward_funcs = [
        SELECTION_REWARD_FUNCS_REGISTRY[func]
        for func in script_args.answer_selection_reward_funcs
    ]

    # Answer-training reward functions (computed on answer-only completion)
    answer_reward_funcs = [SELECTION_REWARD_FUNCS_REGISTRY[func] for func in answer_reward_names]

    # Metric-only reward functions (computed for logging only, on full completion by default)
    metric_reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.metric_reward_funcs]

    dataset = process_dataset(dataset, script_args)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    model_init_kwargs = model_init(model_args, training_args)
    training_args.model_init_kwargs = model_init_kwargs

    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]
    if script_args.train_subset_size is not None:
        train_dataset = train_dataset.select(range(script_args.train_subset_size))
    if script_args.eval_subset_size is not None:
        eval_dataset = eval_dataset.select(range(script_args.eval_subset_size))

    answer_reward_weights = script_args.answer_reward_weights
    if answer_reward_weights is None and script_args.answer_reward_funcs is None:
        # Backward-compatible default: if answer_reward_funcs is not explicitly set, use the
        # same weights as answer selection (previous STPO behavior used selection scores).
        answer_reward_weights = script_args.answer_selection_reward_weights

    trainer = CustomTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=confidence_reward_funcs,
        answer_selection_reward_funcs=answer_selection_reward_funcs,
        answer_selection_reward_weights=script_args.answer_selection_reward_weights,
        answer_reward_funcs=answer_reward_funcs,
        answer_reward_weights=answer_reward_weights,
        metric_reward_funcs=metric_reward_funcs,
        answer_selection_format_pattern=sel_pattern,
        answer_selection_strategy=script_args.answer_selection_strategy,
        answer_selection_balanced_correct_fraction=script_args.answer_selection_balanced_correct_fraction,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
    )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = script_args.train_subset_size
    try:
        trainer.save_state()
    except Exception:
        print("Failed to save state, please debug")
        pass

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {"dataset_name": script_args.dataset_name, "tags": ["rl-verify"]}
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((STPOScriptArguments, STPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
