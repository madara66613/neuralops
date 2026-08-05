from unittest.mock import patch

import pytest
import torch

from neuralops.device import resolve_device


def test_explicit_cpu_is_supported() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("quantum")


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=False)
def test_auto_falls_back_to_cpu(_mps: object, _cuda: object) -> None:
    assert resolve_device("auto") == torch.device("cpu")
