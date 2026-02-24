"""
HRM ACT V2: Transformer Baseline for Architecture Ablation

This is an architecture ablation of the Hierarchical Reasoning Model (HRM).
Key changes from V1:
1. REMOVED hierarchical split (no separate H and L levels)
2. REMOVED inner cycles (no H_cycles/L_cycles loops within reasoning)
3. KEPT ACT outer loop structure intact
4. KEPT all data preprocessing, embeddings, and evaluation infrastructure

Architecture: Single-level transformer that processes the full 30x30 grid as a
900-token sequence, with the same positional encodings and sparse embeddings as V1.

"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class Model_ACTV2InnerCarry:
    z_H: torch.Tensor



class Model_ACTV2Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    grad_cycle_prob: float = 1.0

    H_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_max_steps_eval: Optional[int] = None  # If set, use this for eval instead of halt_max_steps
    halt_exploration_prob: float
    act_enabled: bool = True  # If False, always run halt_max_steps (no early stopping during training)
    act_inference: bool = False  # If True, use adaptive computation during inference
    halt_on_correct: bool = False # If True, halt training only when output matches labels
    halt_on_correct_and_predicted: bool = False # If True, halt when correct AND q_head predicts halt
    halt_train_threshold: float = 0.5 # Halt training if probability of a correct solution is above this threshold
    halt_eval_threshold: float = 0.5 # Halt evaluation if probability of a correct solution is above this threshold
    eval_on_q: bool = False # If True, use ACT-based halting for evaluation
    explore_as_eval: bool = False # If True, explored samples use eval halting strategy
    no_ACT_continue: bool = True # No continue ACT loss, only use the sigmoid of the halt

    forward_dtype: str = "bfloat16"


class Model_ACTV2Block(nn.Module):
    def __init__(self, config: Model_ACTV2Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class Model_ACTV2ReasoningModule(nn.Module):
    def __init__(self, layers: List[Model_ACTV2Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class Model_ACTV2_Inner(nn.Module):
    def __init__(self, config: Model_ACTV2Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = Model_ACTV2ReasoningModule(
            layers=[Model_ACTV2Block(self.config) for _i in range(self.config.H_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return Model_ACTV2InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: Model_ACTV2InnerCarry):
        return Model_ACTV2InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
        )

    def forward(
        self, carry: Model_ACTV2InnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Model_ACTV2InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Random no_grad cycles followed by H_cycles grad steps
        z_H = carry.z_H
        if self.config.grad_cycle_prob < 1.0:
            n_no_grad = np.random.negative_binomial(self.config.H_cycles, self.config.grad_cycle_prob)
            if n_no_grad > 0:
                with torch.no_grad():
                    for _ in range(n_no_grad):
                        z_H = self.H_level(z_H, input_embeddings, **seq_info)
        for _ in range(self.config.H_cycles):
            z_H = self.H_level(z_H, input_embeddings, **seq_info)

        # LM Outputs
        new_carry = Model_ACTV2InnerCarry(
            z_H=z_H.detach(),
        )  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


def Model_ACTV2(config_dict: dict):
    """Factory: creates ACTWrapper around the inner model."""
    from models.act import ACTWrapper
    config = Model_ACTV2Config(**config_dict)
    inner = Model_ACTV2_Inner(config)
    return ACTWrapper(inner, config)
