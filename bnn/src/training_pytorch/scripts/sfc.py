
from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import  *
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Quantized LFC (MNIST) Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
#parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)') #NOT USED
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1234)')
parser.add_argument('--gpus', default=0, help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--resume', default=False, action='store_true', help='Perform only evaluation on val dataset.')
parser.add_argument('--wb', type=int, default=1, metavar='N', choices=[1, 2, 4], help='number of bits for weights (default: 1)')
parser.add_argument('--ab', type=int, default=1, metavar='N', choices=[1, 2, 4],help='number of bits for activations (default: 1)')
parser.add_argument('--eval', default=False, action='store_true', help='perform evaluation of trained model')
parser.add_argument('--export', default=False, action='store_true', help='perform weights export as npz of trained model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
save_path='results/sfc_mnist-w{}a{}.pt'.format(args.wb, args.ab)
neurons = 256
prev_acc = 0

def init_weights(m):
    if type(m) == BinarizeLinear:
        torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)

class lfc(nn.Module):
    def __init__(self):
        super(lfc, self).__init__()

        self.classifier = nn.Sequential(
            Quantizer(1),
            BinarizeLinear(args.wb, 784, neurons, bias=True),
            nn.BatchNorm1d(neurons),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),

            BinarizeLinear(args.wb, neurons, neurons, bias=True),
            nn.BatchNorm1d(neurons),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),

            BinarizeLinear(args.wb, neurons, neurons, bias=True),
            nn.BatchNorm1d(neurons),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),

            BinarizeLinear(args.wb, neurons, 10, bias=True),
            nn.BatchNorm1d(10),
        )
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.classifier(x)
        return x

    def export(self):
        import numpy as np
        dic = {}
        i = 0
        # process linear and BN layers
        for k in range(len(self.classifier)):
            # print(self.classifier[k])
            if hasattr(self.classifier[k], 'weight') and not hasattr(self.classifier[k], 'running_mean'):
                dic['arr_'+str(i)] = np.transpose(self.classifier[k].weight.detach().numpy())
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.classifier[k], 'running_mean'):
                dic['arr_'+str(i)] = self.classifier[k].bias.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = 1./np.sqrt(self.classifier[k].running_var.detach().numpy())
                i = i + 1
        
        save_file = 'results/sfc_mnist-w{}a{}.npz'.format(args.wb, args.ab)
        np.savez(save_file, **dic)
        print("Model exported at: ", save_file)
        
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        # one hot encoding for hinge loss
        target=target.unsqueeze(1)
        if args.cuda:
            target_onehot = torch.cuda.FloatTensor(target.size(0), 10)
        else:
            target_onehot = torch.FloatTensor(target.size(0), 10)
        target_onehot.fill_(-1)
        target_onehot.scatter_(1, target, 1)
        target=target.squeeze()
        target_var = Variable(target_onehot)
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def validate(save_model=False):
    model.eval()
    valid_loss = 0
    correct = 0
    global prev_acc
    with torch.no_grad():
        for data, target in valid_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            valid_loss += criterion(output, target).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    valid_loss /= len(valid_loader.dataset)
    new_acc = 100. * correct.float() / len(valid_loader.dataset)
    print('\Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(valid_loss, correct, len(valid_loader.dataset), new_acc))    
    if new_acc > prev_acc:
        # save model
        if save_model:
            torch.save(model, save_path)
            print("Model saved at: ", save_path, "\n")
        prev_acc = new_acc

def test():
    model.eval()
    test_loss = 0
    correct = 0
    global prev_acc
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    new_acc = 100. * correct.float() / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset), new_acc))

if __name__ == '__main__':
    
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    mnist_transforms = transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])

    train_data = datasets.MNIST('data', train=True,
                              download=True,
                              transform = mnist_transforms)

    test_data = datasets.MNIST('data', train=False,
                              download=True,
                              transform = mnist_transforms)

    n_train_examples = int(len(train_data)*0.9)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = torch.utils.data.random_split(train_data, 
                                                           [n_train_examples, n_valid_examples])

    train_loader = torch.utils.data.DataLoader(train_data, 
                                                 shuffle = True, 
                                                 batch_size=args.batch_size)

    valid_loader = torch.utils.data.DataLoader(valid_data, 
                                                 batch_size=args.batch_size)

    test_loader = torch.utils.data.DataLoader(test_data, 
                                                batch_size=args.test_batch_size)


    model = lfc()
    if args.cuda:
            torch.cuda.set_device(0)
            model.cuda()
    criterion = HingeLoss()
    # test model
    if args.eval:
    	model = torch.load(save_path)
    	test()
    # export npz
    elif args.export:
        model = torch.load(save_path, map_location = 'cpu')
        model.export()
    # train model
    else:
        if args.resume:
            model = torch.load(save_path)
            test()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            validate(save_model=True)
            if epoch%40==0 and epoch != 0:
                new_lr = optimizer.param_groups[0]['lr']*0.1
                print("[INFO] Scaling lr to {}.".format(new_lr))
                optimizer.param_groups[0]['lr']=new_lr

        #Finished Training. Load the best model and test it.
        model = torch.load(save_path)
        test()
