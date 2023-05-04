import torch
from BaselineGAN.Networks import Discriminator

width_per_stage = [3 * x // 4 for x in [1024, 1024, 1024, 1024, 512, 256, 128]]
blocks_per_stage = [2 * x for x in [1, 1, 1, 1, 1, 1, 1]]
cardinality_per_stage = [3 * x for x in [32, 32, 32, 32, 16, 8, 4]]

x = torch.randn(8, 3, 256, 256).to('cuda')
f = Discriminator([*reversed(width_per_stage)], [*reversed(cardinality_per_stage)], [*reversed(blocks_per_stage)], 2).to('cuda')

y = f(x)
print(y)