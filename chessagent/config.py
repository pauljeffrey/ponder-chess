from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    ff_mult: int = 4
    dec_layers: int = 4
    dropout: float = 0.0
    max_line_len: int = 8   # plies predicted by the line decoder
    n_pv: int = 4           # candidate lines per position
    enc_loops: int = 1      # max times the encoder stack (everything above the embedding) is re-applied; no new params
    dec_loops: int = 1      # same for the line decoder
    has_decoder: bool = True  # False: encoder-only model (the deployed form: the look-ahead replays the policy head on a real board)
    loop_embed: bool = False  # True: add a learned per-pass vector (Universal-Transformer style, +d_model*loops params)

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # ~0.3M params: smoke tests on a laptop
    "tiny": dict(d_model=64, n_layers=2, n_heads=4, dec_layers=1),
    # ~4M params: first real run, runs fine on a Mac
    "small": dict(d_model=192, n_layers=6, n_heads=6, dec_layers=2),
    # ~32M params: the target model for the GPU box
    "base": dict(d_model=384, n_layers=12, n_heads=6, dec_layers=4),
    # identical weights to base (same 32.0M params); the stack may be re-applied up to 2x / 3x at the cost of compute
    "base_loop": dict(d_model=384, n_layers=12, n_heads=6, dec_layers=4, enc_loops=2, dec_loops=2),
    # same, plus a learned "which pass am I on" vector per pass (a few thousand extra params)
    "base_loop_emb": dict(d_model=384, n_layers=12, n_heads=6, dec_layers=4, enc_loops=2, dec_loops=2, loop_embed=True),
    # ~64M params: the bigger GPU-box model (wider, deeper encoder and a 5-layer line decoder)
    "large": dict(d_model=512, n_layers=13, n_heads=8, dec_layers=5),
    "tiny_loop": dict(d_model=64, n_layers=2, n_heads=4, dec_layers=1, enc_loops=2, dec_loops=2),
}


def preset(name: str) -> ModelConfig:
    return ModelConfig(**PRESETS[name])


@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    out_dir: str = "runs/default"
    model: str = "tiny"
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    steps: int = 200
    warmup: int = 20
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 500
    num_workers: int = 0
    seed: int = 0
    device: str = "auto"
    w_policy: float = 1.0
    w_value: float = 1.0
    w_line: float = 1.0
    pv_aug: float = 0.0       # share of training positions advanced along their stored best line
    policy_temp_cp: float = 100.0  # softmax temperature (in cp) for turning multi-PV scores into targets
    extra: dict = field(default_factory=dict)
