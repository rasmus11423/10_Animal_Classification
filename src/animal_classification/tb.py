import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


def dummy_training():
    x = torch.randn(3, 3, requires_grad=True)
    y = torch.randn(3, 3)
    z = x @ y
    z.sum().backward()


# Update profiler configuration to use tensorboard_trace_handler
profiler = profile(
    activities=[ProfilerActivity.CPU],  # Add ProfilerActivity.CUDA if GPU is available
    on_trace_ready=tensorboard_trace_handler("runs/profiler_logs"),  # Save logs for TensorBoard
    with_stack=True,
    record_shapes=True,
    profile_memory=True,
)

with profiler:
    for _ in range(5):  # Simulate a training loop
        dummy_training()
        profiler.step()
