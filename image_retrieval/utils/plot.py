import matplotlib.pyplot as plt
import numpy as np


def img_from_tensor(inp, mean, std):
    inp = inp.to("cpu").numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    return inp


def imshow(inp, mean, std, title=None):
    """Imshow for Tensor."""
    inp = img_from_tensor(inp, mean, std)
    plt.imshow(inp)
    plt.axis("off")
