import torch
from torchvision.utils import save_image

def save_test_grid(inputs, samples, save_path, n, width, height):
    inputs = inputs.cpu().data.view(-1, 3, height, width)
    samples = samples.cpu().data.view(-1, 3, height, width)
    images = torch.cat((inputs, samples), dim=1).view(-1, 3, height, width)
    save_image(images, save_path, nrow=n)
