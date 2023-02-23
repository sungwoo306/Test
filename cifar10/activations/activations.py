import torch

e = torch.exp(torch.tensor(1.0))
pi = torch.pi

# simply define a kelu function
def molu(input):
    return input * torch.tanh(2 * torch.exp(2 * input))
