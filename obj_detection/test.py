import torch

i = -2
while i < 2:
    tensor = torch.FloatTensor((1))
    tensor[0] = i
    value = torch.tanh(tensor)
    print(value[0])
    i+=0.2