from pathlib import Path
from weakref import ref
import torch
from src.train.utils import DTYPE_MAP
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import gc

import time
import numpy as np
import html

from src.dataset.countdown_ds import CountdownDataset, collate_fn
from src.models.qwen.qwen2_model import Transformer
from src.models.grpo.memory_efficient_adam import MemoryEfficientAdamW
from src.models.grpo.grpo import (
    single_step_rollout,
    update_policy,
    build_and_load_model,
)
from src.models.grpo.reward import compute_rewards
from src.evals.countdown_eval import evaluate
from src.tokenizers.tokenizer import Tokenizer
from src.train.utils import get_device


def train(config):
    """
    Main training loop for the GRPO algorithm.

    Parameters
    ----------
    config : dict
        A dictionary containing configuration options.
    """

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = get_device()
    dtype = DTYPE_MAP.get(config["model"]["dtype"], torch.bfloat16)
    torch.random.manual_seed(config["training"]["random_seed"])

    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH
    assert (
        NUM_ANSWERS_PER_QUESTION > 0
    ), "BATCH_SIZE must be divisible by NUM_QUESTIONS_PER_BATCH"

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    train_dataset = CountdownDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device="cpu")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    model, optimizer = build_and_load_model(config, device, True)
    ref_model, _ = build_and_load_model(config, device, False)
    # print(torch.cuda.memory_summary(abbreviated=True))

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(train_dataloader, start=1):
        # Samples outputs.
        instances = single_step_rollout(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=compute_rewards,
            device=device,
            dtype=dtype,
            config=config,
        )
        gc.collect()
        torch.cuda.empty_cache()

        if config["training"]["skip_unfinished_instances"]:
            instances = [instance for instance in instances if instance.is_finished]

        # Policy update step.
        results = update_policy(
            model=model,
            optimizer=optimizer,
            instances=instances,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            dtype=dtype,
            config=config,
        )

        # synchronize CUDA to ensure all operations are complete before measuring time
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # Updates the reference model periodically.
        if step % config["training"]["ref_model_update_interval"] == 0:
            # Update the reference model with the current model's state
            ref_model.load_state_dict(model.state_dict())
            print(f"Updated reference model at step {step}")

        # compute and log important metrics
        reward = [instance.reward for instance in instances]
        formatted_reward = [
            instance.reward_info["format_reward"] for instance in instances
        ]
        answer_reward = [
            instance.reward_info["answer_reward"] for instance in instances
        ]
        num_finished_instances = sum(instance.is_finished for instance in instances)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(instance.generated_token_ids) for instance in instances]
        )
        print(
            f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_instances: {num_finished_instances}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
        )

        # logs to TensorBoard
        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(
                model, ref_model, tokenizer, device, dtype, config
            )
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)
        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_instances", num_finished_instances, step)
        tb_writer.add_scalar("learning_rate", lr, step)
        tb_writer.add_scalar("mean_response_len", mean_response_len, step)
        tb_writer.add_scalar("entropy", entropy, step)

        # sort instances by generated token length for easier viz
        instances.sort(key=lambda x: len(x.generated_token_ids))
        for i, instance in enumerate(instances):
            # TensorBoard treats text as markdown.
            text = html.escape(instance.full_text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")
