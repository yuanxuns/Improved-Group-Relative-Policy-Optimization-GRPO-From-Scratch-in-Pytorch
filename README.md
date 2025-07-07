# Improved-Group-Relative-Policy-Optimization-GRPO-From-Scratch-in-Pytorch

GRPO is a powerful reinforcement learning (RL) technique designed to fine-tune large language models (LLMs) by optimizing policies through **group-level comparisons** rather than relying on a separate value function (critic). Initially introduced in **DeepSeekMath** to enhance mathematical reasoning, GRPO underpins advanced LLMs like **DeepSeek‑R1**, **V2**, and **V3**.

![image](https://github.com/user-attachments/assets/7a28d992-21d3-4e27-9a12-a74da5e1f05e)

---

## What Makes GRPO Different?

Traditional policy optimization methods like PPO require a critic network to estimate baseline values, which doubles memory and computation cost, especially for large LLMs. GRPO simplifies this by:

1. Generating group candidate responses per prompt using the policy.
2. Scoring each candidate with a reward model.
3. Computing the mean and standard deviation of these scores.
4. Defining **advantage** as the standardized score:  `A = (r - μ) / σ`.
5. Updating policy via a PPO-style clipped surrogate objective averaged over the group.
6. Optionally include KL divergence regularization to keep the policy close to the supervised model.

![image](https://github.com/user-attachments/assets/04098103-1e23-49f0-a6a1-e944df41765f)

![image](https://github.com/user-attachments/assets/49f19526-30ad-476a-87e4-4aea662769cb)

## Benefits of GRPO

- **Memory-efficient**: No need for large value networks.
- **Stable training**: Normalized advantage and clipping ensure smoother updates.
- **Scalable**: Facilitates training of multi-billion–parameter LLMs like DeepSeek‑R1/V3.
- **Effective**: Proven to enhance mathematical reasoning and coding performance.


## Key Features

- Removes the critic network for efficiency.
- Leverages group-based advantage normalization.
- Compatible with PPO-style clipped objectives.
- Supports KL regularization to maintain alignment with supervised checkpoints.
- Ideal for preference-based or rule-based reward modeling in LLM fine-tuning.
  
## Improved Implementations
- Training QWEN 0.5B on a single Nvidia3060 12GB GPU
- Full parameter tuning
- Token-level policy gradient loss from [DAPO](https://arxiv.org/pdf/2503.14476) paper
- Skips unfinished episodes that exceed context length limits
- Entropy-based mask
- Filtered loss on high-entropy minority tokens from [paper](https://arxiv.org/pdf/2506.01939)
- Memory-efficient AdamW optimizer
- If GRPO iteration is one per training step, we can further simplify the loss.

## Training Results


## References

https://arxiv.org/pdf/2402.03300

https://arxiv.org/pdf/2503.14476

https://github.com/policy-gradient/GRPO-Zero

https://github.com/McGill-NLP/nano-aha-moment

https://github.com/joey00072/nanoGRPO

https://github.com/wyf3/llm_related/blob/main/grpo_from_scratch

