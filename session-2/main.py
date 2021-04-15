import torch
import torch.nn as nn
import torch.optim as optim

from dataset import MyDataset
from model import MyModel
from utils import my_accuracy, save_model

from torch.utils.data import DataLoader
from torchvision import transforms

# Choose GPU if it's available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(train_data, network, optimizer, criterion, epoch):
    # Activate the train=True flag inside the model
    network.train()
    avg_loss = None
    avg_weight = 0.1
    for batch_idx, (sample, code) in enumerate(train_data):
        sample, code = sample.to(device), code.to(device)
        optimizer.zero_grad()
        output = network(sample)
        loss = criterion(output, code)
        loss.backward()
        if avg_loss:
            avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
        else:
            avg_loss = loss.item()
        optimizer.step()
        if batch_idx % config['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_data.dataset),
                100. * batch_idx / len(train_data), loss.item()))
    return avg_loss

def eval_single_epoch(eval_loader, network, criterion):
    network.eval()
    eval_loss = 0
    acc = 0
    with torch.no_grad():
        for sample, code in eval_loader:
            sample, code = sample.to(device), code.to(device)
            output = network(sample)
            eval_loss += criterion(output, code, reduction='sum').item() # sum up batch loss
            # compute number of correct predictions in the batch
            acc += my_accuracy(output, code)
    # Average acc across all correct predictions batches now
    eval_loss /= len(eval_loader.dataset)
    eval_acc = 100. * acc / len(eval_loader.dataset)
    print('\nEval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        eval_loss, acc, len(eval_loader.dataset), eval_acc,
        ))
    return eval_loss, eval_acc

def train_model(config):    
    # Model    
    my_model = MyModel().to(device)
    # LossFunction
    loss = nn.functional.cross_entropy
    # Optimizer
    optimizer = optim.RMSprop(my_model.parameters(), lr=config['learning_rate'])
    for epoch in range(config["epochs"]):
        train_single_epoch(train_loader, my_model, optimizer, loss, epoch)
        eval_single_epoch(eval_loader, my_model, loss)

    return my_model

def test_model(network):
    network.eval()
    acc = 0
    with torch.no_grad():
        for sample, code in test_loader:
            sample, code = sample.to(device), code.to(device)
            output = network(sample)
            # compute number of correct predictions in the batch
            acc += my_accuracy(output, code)
    # Average acc across all correct predictions batches now
    test_acc = 100. * acc / len(test_loader.dataset)
    print('\nNetwork Test: Accuracy: {}/{} ({:.0f}%)\n'.format(
        acc, len(test_loader.dataset), test_acc,
        ))
    return test_acc

if __name__ == "__main__":
    # HyperParameters
    config = {
        'batch_size':64,
        'epochs':10,
        'learning_rate':1e-3,
        'log_interval':20,
    }

    # Define Transform
    my_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
    # Dataset creation
    my_dataset = MyDataset( "C:\DeepLearning\ChineseMnistData\data\data", 
                            "C:\DeepLearning\ChineseMnistData\chinese_mnist.csv", 
                            transform=my_transform)
    train_data, eval_data, test_data = torch.utils.data.random_split(my_dataset, [10000,2500,2500])
    train_loader = torch.utils.data.DataLoader( train_data,
                                                batch_size=config['batch_size'], 
                                                shuffle=True)
    eval_loader = torch.utils.data.DataLoader(  eval_data,
                                                batch_size=config['batch_size'], 
                                                shuffle=False)
    test_loader = torch.utils.data.DataLoader(  test_data,
                                                batch_size=config['batch_size'], 
                                                shuffle=False)

    # Model training                                            
    model = train_model(config)

    print(test_model(model))
    
    save_model(model,"./Models/ModelTest.pth")
