import os

import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from Model import Model


def generate_images(net=Model(),
                    path_to_weigths="checkpoint.pt",
                    predict_n_images=40
                    ):
    net.load_state_dict(torch.load(path_to_weigths, map_location="cpu")["model"])
    os.makedirs("results", exist_ok=True)

    for index in range(predict_n_images):
        image = net()
        image = image.reshape(-1,28,28)
        save_image(image, f'results/{index}.png')


if __name__ == '__main__':
    generate_images()
