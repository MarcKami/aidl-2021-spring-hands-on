import torch

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(train_data, network, optimizer, criterion):
    # Activate the train=True flag inside the model
    network.train()
    avg_loss = None
    avg_weight = 0.1
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        if avg_loss:
            avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
        else:
            avg_loss = loss.item()
        optimizer.step()
        if batch_idx % hparams['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data.dataset),
                100. * batch_idx / len(train_data), loss.item()))
    return avg_loss


def eval_single_epoch():
    pass


def train_model(config):
    
    my_dataset = MyDataset("C:\DeepLearning\ChineseMnistData\data\data", "C:\DeepLearning\ChineseMnistData\chinese_mnist.csv")
    train_data, eval_data, test_data = torch.utils.data.random_split(my_dataset, [10000,2500,2500])
    my_model = MyModel().to(device)
    #for epoch in range(config["epochs"]):
    #    train_single_epoch(...)
    #    eval_single_epoch(...)

    return my_model

def test_model():
    pass


if __name__ == "__main__":

    config = {
        "hyperparam_1": 1,
        "hyperparam_2": 2,
    }
    train_model(config)

    print(test_model())
