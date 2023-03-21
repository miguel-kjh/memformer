
import numpy as np
import torch
import einops

# crete a seed for numpy and torch
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed)

# Rearrange
print('Rearrange')
x = np.random.randn(2, 3, 4, 5)
y = einops.rearrange(x, 'b h w c -> c b h w')
print(x.shape, "->", y.shape)  # (2, 5, 3, 4)

# Reduce
print('Reduce')
x = np.random.randn(2, 3, 4, 5)
y = einops.reduce(x, 'b h w c -> b c', 'min')
print(x.shape, "->", y.shape)  # (2, 3, 4, 5) -> (2, 5)

# Repeat
print('Repeat')
x = np.random.randn(2, 3, 4, 5)
y = einops.repeat(x, 'b h w c -> b h w (c copy)', copy=3)
print(x.shape, "->", y.shape)  # (2, 3, 4, 5) -> (2, 3, 4, 15)