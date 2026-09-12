"""Parameter and FLOPs profiling for the released models.

GFLOPS is measured directly with the torch profiler's FLOPs counter
(``with_flops=True``), which attributes floating-point operations per
operator from the real shapes seen in the forward pass -- no MAC-based
conversion is involved.
"""

import torch


def measure_model_complexity(model, num_points, input_size=128,
                             warmup=1, device='cpu'):
    """Return parameter count and per-crop GFLOPS for one object instance.

    The input spec matches training and evaluation exactly: RGB and organized
    XYZ at ``input_size`` x ``input_size``, a single-channel depth-validity
    map, one class id, and ``num_points`` normalized CAD points.  Profiling
    runs on the CPU (some models hold plain tensor attributes that
    ``.to(device)`` does not move, which breaks GPU tracing).
    """
    from torch.profiler import ProfilerActivity, profile

    original_device = next(model.parameters()).device
    model = model.to('cpu')
    was_training = model.training
    model.eval()
    inputs = (
        torch.zeros(1, 3, input_size, input_size),
        torch.zeros(1, 3, input_size, input_size),
        torch.zeros(1, 1, input_size, input_size),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 3, num_points),
        torch.zeros(1, 1, input_size, input_size))
    with torch.no_grad():
        for _ in range(max(1, int(warmup))):
            model(*inputs)
        with profile(
                activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            model(*inputs)
    flops = sum(event.flops for event in prof.key_averages())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if was_training:
        model.train()
    model.to(original_device)
    return {
        'parameters_M': parameters / 1e6,
        'gflops': flops / 1e9,
        'input_size': int(input_size),
        'num_points': int(num_points),
    }
