import torch

tensor_image = torch.rand(1, 5, 512, 512)
sample_number = 10

sample_grid = torch.randint(0, 512, (sample_number, 3))
sample_grid = torch.index_fill(sample_grid, 1, torch.tensor(0), 0)


# print(sample.shape)  # torch.Size([1, 5, 512, 512])
