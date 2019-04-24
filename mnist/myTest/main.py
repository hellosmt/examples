import torch
import torchvision
from torchvision import transforms
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        pred = self.fc2(x)
        return F.log_softmax(pred, dim=1)


def train(args, model, loader, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label)
        #print(loss.size())  # torch.Size([])  标量  所以后面反向传播时不需要传一个tensor进去
        #print(type(loss))  # <class 'torch.Tensor'>
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}".format(epoch+1, (batch_idx+1)*len(data), len(loader.dataset),
                                                                  100. * batch_idx / len(loader), loss.item()))


def test(args, model, loader, device):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss += F.nll_loss(output, label, reduction='sum').item()  # item()取数值 否则是tensor
            pred = output.argmax(dim=1, keepdim=True)  # keepdim是什么？？
            correct += pred.eq(label.view_as(pred)).sum().item()  # 先把label变成和pred相同的size，然后看和pred哪些元素相等，相等会返回1，不等返回0
    loss /= len(loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-size", type=int, default=4, metavar="--epoch-size", help="input batch size for training (default: 64)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="--momentum", help="SGD momentum (default: 0.5)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="--log-interval", help="how many batches to log")
    parser.add_argument("--save-model", default=False, action='store_true')
    parser.add_argument("--batch-size", type=int, default=64, metavar="--batch-size")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers":1, "pin_memory": True} if use_cuda else {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1235,), (0.3458,))])
    dataset_train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataset_test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

    trainLoader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size, **kwargs)
    testLoader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=args.momentum)

    for epoch in range(args.epoch_size):
        train(args, model, trainLoader, optimizer, epoch, device)
        test(args, model, testLoader, device)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist.pt")

if __name__ == '__main__':
    main()