import argparse
import torch
import torchvision
from torchvision import transforms

# parser = argparse.ArgumentParser()
# parser.add_argument("integer", type=int, help="display a int")
# args = parser.parse_args()
# print(args.integer)

train_datasets = torchvision.datasets.MNIST("../data")
print(train_datasets)
print(type(train_datasets))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1327,), (0.3456,))]
)
train_datasets_v2 = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
print(train_datasets_v2)
print(type(train_datasets_v2))