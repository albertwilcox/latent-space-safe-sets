import torch
import numpy as np


TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def torchify(x):
    if type(x) is not torch.Tensor and type(x) is not np.ndarray:
        x = np.array(x)
    if type(x) is not torch.Tensor:
        x = torch.FloatTensor(x)
    return x.to(TORCH_DEVICE)


def to_numpy(x):
    if x is None:
        return x
    return x.detach().cpu().numpy()
