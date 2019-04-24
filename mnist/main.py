from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)  # nn.Linear负责构建全连接层，需要提供输入和输出的通道数

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # Sets the module in training mode.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # to 方法可以轻松的将目标从一个设备转移到另一个设备(比如从 cpu 到 cuda )
        optimizer.zero_grad()
        # 调用x.__call__(1,2)等同于调用x(1,2)
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播, Python中类的实例（对象）可以被当做函数对待,为了将一个类实例当做函数调用，我们需要在类中实现__call__()方法
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  # /t：横向制表符
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss 侯爱民的sum是这一个batch的损失的sum
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # # 先把label变成和pred相同的size，然后看和pred哪些元素相等，相等会返回1，不等返回0
            correct += pred.eq(target.view_as(pred)).sum().item()  # 返回被视作与给定的tensor相同大小的原tensor。 等效于：self.view(tensor.size())

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    # 创建 ArgumentParser() 对象
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # 添加参数  metavar:帮助信息中显示的参数名称
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',  # 训练多少个batch打印一下loss
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    # 解析参数
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    #  num_workers:how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
    #  pin_memory:锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # 通过DataLoader返回一个数据集上的可迭代对象。一会我们通过for循环，就可以遍历数据集了
    train_loader = torch.utils.data.DataLoader(
        # MNIST是torchvision.datasets包中的一个类，负责根据传入的参数加载数据集。如果自己之前没有下载过该数据集，
        # 可以将download参数设置为True，会自动下载数据集并解包。如果之前已经下载好了，只需将其路径通过root传入即可。
        datasets.MNIST('../data', train=True, download=True,
                       #  list of transforms to compose 对数据的预处理
                       transform=transforms.Compose([
                           # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor 范围[0,1]
                           transforms.ToTensor(),
                           # 标准化 Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
                           #     will normalize each channel of the input ``torch.*Tensor`` i.e.
                           #     ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
                           transforms.Normalize((0.1307,), (0.3081,))  # 只有一个通道，传入tuple
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)  # to(): Moves and/or casts the parameters and buffers.
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        # model.state_dict() 只保存网络中的参数 (速度快, 占内存少) model_object.load_state_dict(torch.load('params.pkl'))
        # 如果是torch.save(model, "mnist_cnn.pt")则是保存整个网络 在恢复时不需要重建网络构造  model = torch.load('model.pkl')
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
