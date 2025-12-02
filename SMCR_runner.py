import logging
import os
import sys
from functools import partial

import datasets
import torch
import transformers
from arguments import GRPOConfig, GRPOScriptArguments, ModelConfig
from dataset_processing import process_dataset
from datasets import load_dataset
from reward_fns import (
    accuracy_reward,
    brier_reward,
    confidence_one_or_zero,
    format_reward,
    mean_confidence_reward,
)
from SMCR_Trainer import CustomTrainer
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

    # Log on each process a small summary
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

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "format": partial(format_reward, format_pattern=script_args.format_pattern),
        "accuracy": partial(accuracy_reward, format_pattern=script_args.format_pattern),
        "brier": partial(brier_reward, format_pattern=script_args.format_pattern),
        "mean_confidence": mean_confidence_reward,
        "confidence_one_or_zero": confidence_one_or_zero,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

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

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = CustomTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = script_args.train_subset_size
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    try:
        trainer.save_state()
    except:
        print("Failed to save state, please debug")
        pass

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["rl-verify"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
