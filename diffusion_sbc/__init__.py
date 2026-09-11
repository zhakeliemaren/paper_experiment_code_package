"""Runnable prototype for diffusion-driven stochastic verification."""

from .config import ExperimentConfig
from .pipeline import train_pipeline

__all__ = ["ExperimentConfig", "train_pipeline"]
