"""
Shared ACT (Adaptive Computation Time) outer loop for all recursive reasoning models.

All models share the same carry management, halting logic, and data recycling.
The inner model just needs to implement:
  - empty_carry(batch_size) -> inner_carry
  - reset_carry(mask, inner_carry) -> inner_carry
  - forward(inner_carry, data) -> (inner_carry, logits, (q_halt_logits, q_continue_logits))
"""

from typing import Tuple, Dict, Any
from dataclasses import dataclass

import torch
from torch import nn

IGNORE_LABEL_ID = -100


@dataclass
class ACTCarry:
    inner_carry: Any
    steps: torch.Tensor       # [batch_size] int32
    halted: torch.Tensor      # [batch_size] bool
    current_data: Dict[str, torch.Tensor]
    use_eval_halting: torch.Tensor  # [batch_size] bool, persisted across steps


class ACTWrapper(nn.Module):
    """Shared ACT wrapper for all recursive reasoning models."""

    def __init__(self, inner_model: nn.Module, config):
        super().__init__()
        self.inner = inner_model
        self.config = config

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch_size: int, sample_template: Dict[str, torch.Tensor]) -> ACTCarry:
        return ACTCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={
                k: v.unsqueeze(0).expand(batch_size, *v.shape).clone()
                for k, v in sample_template.items()
            },
            use_eval_halting=torch.zeros((batch_size,), dtype=torch.bool),
        )

    def forward(
        self,
        carry: ACTCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[ACTCarry, Dict[str, torch.Tensor]]:
        # Reset carry for halted positions
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            device = new_steps.device

            # Per-sample halting strategy (persisted in carry, refreshed for new samples)
            if self.training and self.config.explore_as_eval:
                new_rolls = torch.rand(new_steps.shape[0], device=device) < self.config.halt_exploration_prob
                use_eval_halting = torch.where(carry.halted, new_rolls, carry.use_eval_halting)
            elif not self.training:
                use_eval_halting = torch.ones(new_steps.shape[0], dtype=torch.bool, device=device)
            else:
                use_eval_halting = torch.zeros(new_steps.shape[0], dtype=torch.bool, device=device)

            # Derived per-sample parameters
            eval_max_steps = self.config.halt_max_steps_eval if self.config.halt_max_steps_eval is not None else self.config.halt_max_steps
            effective_max_steps = torch.where(use_eval_halting, eval_max_steps, self.config.halt_max_steps)
            effective_threshold = torch.where(use_eval_halting, self.config.halt_eval_threshold, self.config.halt_train_threshold)

            is_last_step = new_steps >= effective_max_steps
            halted = is_last_step
            outputs["is_last_step"] = is_last_step

            # Check if adaptive computation should be used
            act_enabled = self.config.act_enabled
            act_inference = self.config.act_inference
            eval_on_q = self.config.eval_on_q
            no_ACT_continue = self.config.no_ACT_continue

            use_adaptive = (self.config.halt_max_steps > 1) and (
                (self.training and act_enabled)
                or (not self.training and (act_inference or eval_on_q))
            )

            if use_adaptive:
                q_halt_prob = torch.sigmoid(q_halt_logits)

                # Halt signal
                halt_on_correct = self.config.halt_on_correct
                halt_on_correct_and_predicted = self.config.halt_on_correct_and_predicted

                if (halt_on_correct_and_predicted or halt_on_correct) and self.training:
                    preds = logits.argmax(-1)
                    labels = new_current_data["labels"]
                    correct = ((preds == labels) | (labels == IGNORE_LABEL_ID)).all(dim=-1)
                    if halt_on_correct_and_predicted:
                        oracle_halt = correct & (q_halt_prob >= effective_threshold)
                    else:
                        oracle_halt = correct
                    # Eval-path explored samples use q-based halting, not oracle
                    if no_ACT_continue:
                        eval_halt = q_halt_prob >= effective_threshold
                    else:
                        eval_halt = q_halt_logits > q_continue_logits
                    halted = halted | torch.where(use_eval_halting, eval_halt, oracle_halt)
                elif no_ACT_continue:
                    halted = halted | (q_halt_prob >= effective_threshold)
                else:
                    q_halt_signal = q_halt_logits > q_continue_logits
                    halted = halted | q_halt_signal

                # Store actual steps used for logging (only during inference)
                if not self.training:
                    outputs["actual_steps"] = new_steps.float()

                # Exploration (training only, when NOT using explore_as_eval)
                # NOTE: re-rolls every forward pass, so exploration is weaker than intended.
                if self.training and not self.config.explore_as_eval:
                    min_halt_steps = (
                        torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                    ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q (training only, when using Q-continue)
                if self.training and not no_ACT_continue:
                    next_q_halt_logits, next_q_continue_logits = self.inner(
                        new_inner_carry, new_current_data
                    )[-1]
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return ACTCarry(new_inner_carry, new_steps, halted, new_current_data, use_eval_halting), outputs
