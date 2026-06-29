from types import SimpleNamespace

import pytest
import torch

import sys
from pathlib import Path
import types

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "reformer_pytorch" not in sys.modules:
    reformer_module = types.ModuleType("reformer_pytorch")

    class LSHSelfAttention:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("LSHSelfAttention is not used in these PatchTST compatibility tests.")

    reformer_module.LSHSelfAttention = LSHSelfAttention
    sys.modules["reformer_pytorch"] = reformer_module

if "einops" not in sys.modules:
    einops_module = types.ModuleType("einops")

    def rearrange(tensor, *args, **kwargs):
        del args, kwargs
        return tensor

    def repeat(tensor, *args, **kwargs):
        del args, kwargs
        return tensor

    einops_module.rearrange = rearrange
    einops_module.repeat = repeat
    sys.modules["einops"] = einops_module

from models.FreTS import Model as FreTSModel
from models.PatchTST import Model as PatchTSTModel


def _patchtst_args(**overrides):
    base = {
        "task_name": "long_term_forecast",
        "seq_len": 384,
        "pred_len": 384,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 1,
        "d_model": 32,
        "dropout": 0.0,
        "factor": 1,
        "n_heads": 4,
        "e_layers": 1,
        "d_ff": 64,
        "activation": "gelu",
        "patch_len": 16,
        "stride": 8,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _frets_args(**overrides):
    base = {
        "task_name": "long_term_forecast",
        "seq_len": 384,
        "pred_len": 384,
        "enc_in": 7,
        "channel_independence": 1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_model_inputs(batch_size=1, seq_len=384, pred_len=384, channels=7):
    batch_x = torch.randn(batch_size, seq_len, channels, dtype=torch.float32)
    batch_x_mark = torch.zeros(batch_size, seq_len, 5, dtype=torch.float32)
    batch_y = torch.zeros(batch_size, pred_len, channels, dtype=torch.float32)
    batch_y_mark = torch.zeros(batch_size, pred_len, 5, dtype=torch.float32)
    return batch_x, batch_x_mark, batch_y, batch_y_mark


def _run_forward_backward(model):
    batch_x, batch_x_mark, batch_y, batch_y_mark = _make_model_inputs()
    output = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    assert output.shape == (1, 384, 7)

    ms_output = output[:, :, -1:]
    assert ms_output.shape == (1, 384, 1)

    target = torch.zeros_like(output)
    loss = torch.mean((output - target) ** 2)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)


def test_patchtst_default_patch_len_and_stride_forward_backward():
    model = PatchTSTModel(_patchtst_args())
    _run_forward_backward(model)


def test_patchtst_pchip15_scale_patch_len_and_stride_forward_backward():
    model = PatchTSTModel(_patchtst_args(patch_len=64, stride=32))
    _run_forward_backward(model)


def test_frets_384_to_384_forward_backward():
    model = FreTSModel(_frets_args())
    _run_forward_backward(model)


@pytest.mark.parametrize(
    "patch_len,stride,error_pattern",
    [
        (0, 8, "patch_len must be > 0"),
        (16, 0, "stride must be > 0"),
        (385, 8, "patch_len must be <= seq_len"),
    ],
)
def test_patchtst_invalid_patch_configuration_fails_fast(patch_len, stride, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        PatchTSTModel(_patchtst_args(patch_len=patch_len, stride=stride))
