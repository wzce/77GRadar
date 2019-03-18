import torch

a = torch.randn(3, 3)

print('a: ', a)
a_max = torch.max(a, 1)

print('a_max: ', a_max)
