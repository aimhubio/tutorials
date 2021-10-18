import os
import torch
import operator
import importlib
from packaging.version import Version
from pkg_resources import DistributionNotFound
from transformers import set_seed


def compare_version(package: str, op, version) -> bool:
    """
    Compare package version with some requirements

    >>> compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = importlib.import_module(package)
    except (ModuleNotFoundError, DistributionNotFound):
        return False
    try:
        pkg_version = Version(pkg.__version__)
    except TypeError:
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, Version(version))


_TORCH_GREATER_EQUAL_1_7 = compare_version("torch", operator.ge, "1.7.0")
_TORCH_GREATER_EQUAL_1_8 = compare_version("torch", operator.ge, "1.8.0")


def set_random_state(seed, deterministic):
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # modify from:
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/afbf703684f9996fefce301e4594afa31ba7975f/pytorch_lightning/trainer/connectors/accelerator_connector.py#L190
    if _TORCH_GREATER_EQUAL_1_8:
        torch.use_deterministic_algorithms(deterministic)
    elif _TORCH_GREATER_EQUAL_1_7:
        torch.set_deterministic(deterministic)
    else:  # the minimum version Lightning supports is PyTorch 1.6
        torch._set_deterministic(deterministic)
    if deterministic:
        # fixing non-deterministic part of horovod
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
        os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
