"""Backwards-compatible alias for :mod:`synther.diffusion.diffusion`.

This module used to be a byte-identical copy of `diffusion.py`, while the Ours entry
point imported from `diffusion` and the baseline from `diffusion_cond` -- so a fix
applied to one file silently left the other method running different code
(exp/02_code_notes.md I-13). It is now a re-export; edit `diffusion.py` only.
"""
from synther.diffusion.diffusion import *  # noqa: F401,F403
from synther.diffusion.diffusion import (  # noqa: F401
    DiffusionModel,
    QFlow,
    QFlowMLP,
    ResidualMLP,
    ResidualBlock,
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
)
