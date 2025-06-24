import torch
#torch.cuda.empty_cache()


print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("Cached:   ", torch.cuda.memory_reserved() / 1024**2, "MB")