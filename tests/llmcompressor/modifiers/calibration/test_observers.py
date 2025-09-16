import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    initialize_module_for_quantization,
    QuantizationStatus,
)

from llmcompressor.modifiers.quantization.calibration import (
    initialize_observer,
    update_weight_zp_scale,
    update_weight_global_scale,
    calibrate_input_hook,
)


@pytest.mark.parametrize(
    "shape,group_size,actorder",
    [
        ((1, 1), None, False),
        ((1, 1), 128, False),
        ((1, 1), 128, True),
        ((64, 64), None, False),
        ((64, 64), 128, False),
        ((64, 64), 128, True),
        ((1792, 4096), None, False),
        ((1792, 4096), 128, False),
        ((1792, 4096), 128, True),
        ((3420, 64), None, False),
        ((3420, 64), 128, False),
        ((3420, 64), 128, True),
    ],
)
def test_observers_update(shape, group_size, actorder):
    module = torch.nn.Linear(*shape)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(group_size=group_size, actorder=actorder),
        input_activations=QuantizationArgs(),
        output_activations=QuantizationArgs(),
    )

    input = torch.empty(module.in_features, dtype=module.weight.dtype)
    output = torch.empty(module.out_features, dtype=module.weight.dtype)

    initialize_module_for_quantization(module, scheme)
    initialize_observer(module, "weight")
    initialize_observer(module, "input")
    initialize_observer(module, "output")

    for location, value in (
        ("weight", module.weight),
        ("input", input),
        ("output", output),
    ):
        observer = getattr(module, f"{location}_observer")
        g_idx = getattr(module, "g_idx", None)
        updated_scale, updated_zero_point = observer(value, g_idx=g_idx)

        assert_alike(updated_scale, getattr(module, f"{location}_scale"))
        assert_alike(updated_zero_point, getattr(module, f"{location}_zero_point"))


def assert_alike(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape


@pytest.mark.parametrize(
    "args,exp_min_val,exp_max_val,exp_tol",
    [
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="tensor",
                observer="minmax",
            ),
            {"default": torch.tensor(0.0, dtype=torch.bfloat16)},
            {"default": torch.tensor(23.0, dtype=torch.bfloat16)},
            2.5,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="channel",
                observer="minmax",
            ),
            {"default": torch.tensor([[0], [6], [12], [18]], dtype=torch.bfloat16)},
            {"default": torch.tensor([[5], [11], [17], [23]], dtype=torch.bfloat16)},
            2.5,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=3,
                observer="minmax",
            ),
            {
                "default": torch.tensor([[0], [6], [12], [18]], dtype=torch.bfloat16),
                1: torch.tensor([[3], [9], [15], [21]], dtype=torch.bfloat16),
            },
            {
                "default": torch.tensor([[2], [8], [14], [20]], dtype=torch.bfloat16),
                1: torch.tensor([[5], [11], [17], [23]], dtype=torch.bfloat16),
            },
            2.5,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="float",
                symmetric=True,
                strategy="tensor_group",
                group_size=3,
                observer="minmax",
            ),
            {
                "default": torch.tensor([[0], [6], [12], [18]], dtype=torch.bfloat16),
                1: torch.tensor([[3], [9], [15], [21]], dtype=torch.bfloat16),
            },
            {
                "default": torch.tensor([[2], [8], [14], [20]], dtype=torch.bfloat16),
                1: torch.tensor([[5], [11], [17], [23]], dtype=torch.bfloat16),
            },
            5.0,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="block",
                block_structure=[2, 3],
                observer="minmax",
            ),
            {
                "block_0_0": torch.tensor([[0]], dtype=torch.bfloat16),
                "block_0_1": torch.tensor([[3]], dtype=torch.bfloat16),
                "block_1_0": torch.tensor([[12]], dtype=torch.bfloat16),
                "block_1_1": torch.tensor([[15]], dtype=torch.bfloat16),
            },
            {
                "block_0_0": torch.tensor([[8]], dtype=torch.bfloat16),
                "block_0_1": torch.tensor([[11]], dtype=torch.bfloat16),
                "block_1_0": torch.tensor([[20]], dtype=torch.bfloat16),
                "block_1_1": torch.tensor([[23]], dtype=torch.bfloat16),
            },
            2.5,
        ),
    ],
)
def test_static_weight_quantization(args, exp_min_val, exp_max_val, exp_tol):
    # set up weight
    input_size, output_size = 6, 4
    linear = torch.nn.Linear(input_size, output_size, bias=False)
    linear.weight.data = torch.arange(
        input_size * output_size, dtype=torch.bfloat16
    ).reshape(output_size, input_size)

    # initialize quantization parameters
    scheme = QuantizationScheme(targets=[], weights=args)
    initialize_module_for_quantization(linear, scheme)
    assert getattr(linear, "quantization_scheme") is scheme

    # calibrate quantization parameters
    initialize_observer(linear, "weight")
    update_weight_global_scale(linear)
    update_weight_zp_scale(linear)

    observer = getattr(linear, "weight_observer")
    assert (
        observer.min_val.keys()
        == observer.max_val.keys()
        == exp_min_val.keys()
        == exp_max_val.keys()
    )
    for key in observer.min_val.keys():
        assert torch.equal(observer.min_val[key], exp_min_val[key])
        assert torch.equal(observer.max_val[key], exp_max_val[key])

    # forward pass
    input = torch.rand((1, input_size), dtype=torch.bfloat16)
    output = linear(input)
    true_output = input @ linear.weight.T
    assert torch.allclose(output, true_output, atol=exp_tol)


@pytest.mark.parametrize(
    "args,exp_min_val,exp_max_val,exp_tol",
    [
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="tensor",
                observer="minmax",
            ),
            {"default": torch.tensor(0.0, dtype=torch.bfloat16)},
            {"default": torch.tensor(5.0, dtype=torch.bfloat16)},
            2.5,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="token",
                observer="minmax",
            ),
            {"default": torch.tensor(0.0, dtype=torch.bfloat16)},
            {"default": torch.tensor(5.0, dtype=torch.bfloat16)},
            2.5,
        ),
        # channel is not supported, but (tensor == token == channel)
        # (
        #     QuantizationArgs(
        #         num_bits=4,
        #         type="int",
        #         symmetric=True,
        #         strategy="channel",
        #         observer="minmax",
        #     ),
        #     {"default": torch.tensor(0.0, dtype=torch.bfloat16)},
        #     {"default": torch.tensor(5.0, dtype=torch.bfloat16)},
        #     2.5,
        # ),
        (
            QuantizationArgs(
                num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=3,
                observer="minmax",
            ),
            {
                "default": torch.tensor([[0]], dtype=torch.bfloat16),
                1: torch.tensor([[3]], dtype=torch.bfloat16),
            },
            {
                "default": torch.tensor([[2]], dtype=torch.bfloat16),
                1: torch.tensor([[5]], dtype=torch.bfloat16),
            },
            2.5,
        ),
        (
            QuantizationArgs(
                num_bits=4,
                type="float",
                symmetric=True,
                strategy="tensor_group",
                group_size=3,
                observer="minmax",
            ),
            {
                "default": torch.tensor([[0]], dtype=torch.bfloat16),
                1: torch.tensor([[3]], dtype=torch.bfloat16),
            },
            {
                "default": torch.tensor([[2]], dtype=torch.bfloat16),
                1: torch.tensor([[5]], dtype=torch.bfloat16),
            },
            2.5,
        ),
        # (
        #     QuantizationArgs(
        #         num_bits=4,
        #         type="int",
        #         symmetric=True,
        #         strategy="block",
        #         block_structure=[2, 3],
        #         observer="minmax",
        #     ),
        #     {
        #         "block_0_0": torch.tensor([[0]], dtype=torch.bfloat16),
        #         "block_0_1": torch.tensor([[3]], dtype=torch.bfloat16),
        #         "block_1_0": torch.tensor([[12]], dtype=torch.bfloat16),
        #         "block_1_1": torch.tensor([[15]], dtype=torch.bfloat16),
        #     },
        #     {
        #         "block_0_0": torch.tensor([[8]], dtype=torch.bfloat16),
        #         "block_0_1": torch.tensor([[11]], dtype=torch.bfloat16),
        #         "block_1_0": torch.tensor([[20]], dtype=torch.bfloat16),
        #         "block_1_1": torch.tensor([[23]], dtype=torch.bfloat16),
        #     },
        #     2.5,
        # ),
    ],
)
def test_static_activation_quantization(args, exp_min_val, exp_max_val, exp_tol):
    # set up activation (and identity weight)
    input_size = 6
    input = torch.arange(input_size, dtype=torch.bfloat16).unsqueeze(0)
    linear = torch.nn.Linear(input_size, input_size, bias=False)
    linear.weight.data = torch.eye(input_size, dtype=torch.bfloat16)

    # initialize quantization parameters
    scheme = QuantizationScheme(targets=[], input_activations=args)
    initialize_module_for_quantization(linear, scheme)
    assert getattr(linear, "quantization_scheme") is scheme

    # calibrate quantization parameters
    initialize_observer(linear, "input")
    linear.register_forward_pre_hook(calibrate_input_hook)
    
    # calibration forward pass
    output = linear(input)
    assert torch.allclose(output, input, atol=exp_tol)

    # check calibration
    observer = getattr(linear, "input_observer")
    breakpoint()
    assert (
        observer.min_val.keys()
        == observer.max_val.keys()
        == exp_min_val.keys()
        == exp_max_val.keys()
    )
    for key in observer.min_val.keys():
        assert torch.equal(observer.min_val[key], exp_min_val[key])
        assert torch.equal(observer.max_val[key], exp_max_val[key])