"""Micro-Gemma-FP configuration — ~164M param model for FP8/FP4 experiments.

Architecture preserves Gemma 4 features (RMSNorm, RoPE, GQA, sliding/full
attention, per-layer token embeddings) at a scale suitable for meaningful
quantization research: 12 layers × 768 hidden × 12 heads.

The deeper architecture (12 vs 6 layers) provides non-trivial per-layer
sensitivity for mixed-precision experiments, and larger weight matrices
(768×768) benefit from GPTQ column compensation.
"""

from dataclasses import dataclass, field


@dataclass
class MicroGemmaFPConfig:
    # ═══════════════════════════════════════════════════════════════
    # Architecture — ~164M params
    # ═══════════════════════════════════════════════════════════════
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 3       # GQA ratio 4:1
    head_dim: int = 64
    global_head_dim: int = 128         # Full attention uses 2× head dim

    # Gemma 4 signature: per-layer token embeddings
    hidden_size_per_layer_input: int = 64

    # Vocabulary & position
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    sliding_window: int = 256

    # Layer type distribution (preserves Gemma 4's alternating pattern)
    # 8 sliding (local) + 4 full (global) = 12 layers
    layer_types: list = field(default_factory=lambda: [
        'sliding', 'sliding', 'full',
        'sliding', 'sliding', 'full',
        'sliding', 'sliding', 'full',
        'sliding', 'sliding', 'full',
    ])

    # Normalization & activations (matches Gemma 4)
    rms_norm_eps: float = 1e-6
    hidden_activation: str = 'gelu_pytorch_tanh'

    # Attention
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0

    # ═══════════════════════════════════════════════════════════════
    # Training defaults
    # ═══════════════════════════════════════════════════════════════
    max_seq_length: int = 512
    initializer_range: float = 0.02

    # ═══════════════════════════════════════════════════════════════
    # Quantization flags (override per experiment)
    # ═══════════════════════════════════════════════════════════════
    quantize_weights: str = 'none'       # 'none' | 'fp8' | 'fp4'
    quantize_activations: str = 'none'   # 'none' | 'fp8' | 'fp4'
    stochastic_rounding: bool = False
    hadamard_rotation: bool = False

    # Condition number regularization
    lambda_cond: float = 0.0  # 0 = disabled, recommended: 1e-4 to 1e-3

    # ═══════════════════════════════════════════════════════════════
    # Derived properties
    # ═══════════════════════════════════════════════════════════════

    @property
    def num_sliding_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == 'sliding')

    @property
    def num_full_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == 'full')

    @property
    def model_name(self) -> str:
        """Descriptive name based on quantization config."""
        parts = ['gemma_fp_164m']
        if self.quantize_weights != 'none':
            parts.append(f'w{self.quantize_weights}')
        if self.quantize_activations != 'none':
            parts.append(f'a{self.quantize_activations}')
        if self.stochastic_rounding:
            parts.append('sr')
        if self.hadamard_rotation:
            parts.append('hadamard')
        if self.lambda_cond > 0:
            parts.append(f'cond{self.lambda_cond:.0e}')
        return '_'.join(parts)
