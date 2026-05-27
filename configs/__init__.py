from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import os


def load_config(model_name: str = None, overrides: list = None):
    """Load config with override chain: default -> model config -> argparse overrides.
    
    Args:
        model_name: Model name to load specific config (e.g., 'neumf', 'sasrec')
        overrides: List of override strings from argparse (e.g., ['batch=128', 'lr=0.01'])
    
    Returns:
        DictConfig with merged configuration
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_dir = os.path.join(os.path.dirname(__file__))
    initialize_config_dir(config_dir=config_dir, version_base=None)

    defaults = []
    if model_name:
        defaults.append(f"model={model_name}")

    cfg = compose(config_name="config", overrides=defaults + (overrides or []))
    return cfg


__all__ = ['load_config']
