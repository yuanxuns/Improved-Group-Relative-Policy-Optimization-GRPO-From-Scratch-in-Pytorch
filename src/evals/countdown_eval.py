from torch.utils.data import DataLoader
import torch
import numpy as np
import gc

from src.models.grpo.grpo import single_step_rollout
from src.dataset.countdown_ds import CountdownDataset, collate_fn
from src.models.grpo.reward import compute_rewards


def evaluate(model, ref_model, tokenizer, device, dtype, config):
    """
    Evaluate the model on the test dataset.

    Parameters
    ----------
    model : Transformer
        The model to evaluate.
    ref_model : Transformer
        The reference model to use for the rollout.
    tokenizer : Tokenizer
        The tokenizer to use for the evaluation.
    device : torch.device
        The device to use for the evaluation.
    dtype : torch.dtype
        The data type to use for the evaluation.
    config : dict
        The configuration dictionary.

    Returns
    -------
    mean_reward : float
        The mean reward of the model on the test dataset.
    """
    test_dataset = CountdownDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device="cpu")
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    ref_model.to(device)
    ref_model.eval()
    model.eval()
    for batch in dataloader:
        episodes = single_step_rollout(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_function=compute_rewards,
            device=device,
            dtype=dtype,
            config=config,
        )
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    ref_model.cpu()
    model.train()

    gc.collect()
    torch.cuda.empty_cache()

    return np.mean(success)
