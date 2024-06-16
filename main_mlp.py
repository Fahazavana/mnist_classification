import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from train_models import train_mlp
from models import ClassifierMLP
from torch.utils.tensorboard import SummaryWriter

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
BATCH_SIZE = 256
LATENT_DIM = 100
N_EPOCHS = 1
LR = 2e-3
transforms = Compose([ToTensor()])
mnist_train = MNIST(root='../ccn_gan_torch/train', train=True,
                    download=False, transform=transforms)
mnist_test = MNIST(root='../ccn_gan_torch/test', train=False,
                   download=False, transform=transforms)

train = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
test = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True)

model = ClassifierMLP(28*28, 10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
history = {'loss': [], 'd_loss': []}

writer = SummaryWriter(f"runs/MLP")

train_mlp(model, criterion, optimizer, 10, train, device, writer, test)

