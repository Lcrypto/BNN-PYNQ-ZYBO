# added possibility to test a single image (used to test the DEER image of the finn testbench to assess correct weight loading in the network.


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
from binarized_modules import *
import numpy as np

EXPERIMENT = "[VERSION INFO] 17-MAY-2020. hcnv model, with validation test, scales lr every 40 epochs. Can use with FINN."

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Quantized CNV (Cifar10) Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
#parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)') #NOT USED
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1234)')
parser.add_argument('--gpus', default=0, help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--resume', default=False, action='store_true', help='Perform only evaluation on val dataset.')
parser.add_argument('--wb', type=int, default=1, metavar='N', choices=[1, 2], help='number of bits for weights (default: 1)')
parser.add_argument('--ab', type=int, default=1, metavar='N', choices=[1, 2], help='number of bits for activations (default: 1)')
parser.add_argument('--eval', default=False, action='store_true', help='perform evaluation of trained model')
parser.add_argument('--export', default=False, action='store_true', help='perform weights export as npz of trained model')
parser.add_argument('--test_on_image', action = 'store', help='perform evaluation on a single image. Use for example on - deer.bin -')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
save_path='./results/hcnv_cifar10-w{}a{}.pt'.format(args.wb, args.ab)
prev_acc = 0
def init_weights(m):
    if type(m) == BinarizeLinear or type(m) == BinarizeConv2d:
        torch.nn.init.uniform_(m.weight, -1, 1)
        #The last layer doesn't have bias.
        if(m.bias!=None):
          m.bias.data.fill_(0.01)

class cnv(nn.Module):
    def __init__(self):
        super(cnv, self).__init__()

        self.features = nn.Sequential(
            #input image: 32x32x3
            BinarizeConv2d(args.wb, 3, 16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),
            #input to maxpool: 28x28x16
            nn.MaxPool2d(kernel_size=2, stride=2),

            #input image: 14x14x16
            BinarizeConv2d(args.wb, 16, 32, kernel_size=5, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),
            #input to maxpool: 10x10x32
            nn.MaxPool2d(kernel_size=2, stride=2),

            #input image: 5x5x32
            BinarizeConv2d(args.wb, 32, 64, kernel_size=5, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),
            )
        #input FLAT: 1x1x64
        self.classifier = nn.Sequential(
            BinarizeLinear(args.wb, 64*1, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),

            BinarizeLinear(args.wb, 128, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.Hardtanh(inplace=True),
            Quantizer(args.ab),

            BinarizeLinear(args.wb, 128, 10, bias=False),
            nn.BatchNorm1d(10),
            nn.LogSoftmax()
        )
        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x):
        #print("[DEBUG] x in shape: {}".format(x.shape))
        #print("[DEBUG] type: {}".format(x.device))
        x = self.features(x)
        x = x.view(-1, 64)
        x = self.classifier(x)
        #print("[DEBUG] x out shape: {}".format(x.shape))
        return x

    def export(self):
        import numpy as np
        dic = {}
        i = 0
        
        # process conv and BN layers
        for k in range(len(self.features)):
            if hasattr(self.features[k], 'weight') and not hasattr(self.features[k], 'running_mean'):
                dic['arr_'+str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.features[k], 'running_mean'):
                dic['arr_'+str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = 1./np.sqrt(self.features[k].running_var.detach().numpy())
                i = i + 1
        
        # process linear and BN layers
        for k in range(len(self.classifier)):
            if hasattr(self.classifier[k], 'weight') and not hasattr(self.classifier[k], 'running_mean'):
                dic['arr_'+str(i)] = np.transpose(self.classifier[k].weight.detach().numpy())
                i = i + 1
                if(self.classifier[k].bias != None):
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
        
        save_file = './results/hcnv_cifar10-w{}a{}.npz'.format(args.wb, args.ab)
        np.savez(save_file, **dic)
        print("Model exported at: ", save_file)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
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
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(valid_loss, correct, len(valid_loader.dataset), new_acc))    
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

    # Indentify
    print(EXPERIMENT)
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Load dataset and define transforms.
    train_data = datasets.CIFAR10(root = 'data', 
                                  train = True, 
                                  download = True)

    means = train_data.data.mean(axis = (0,1,2)) / 255
    stds = train_data.data.std(axis = (0,1,2)) / 255

    train_transforms = transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(10),
                               transforms.RandomCrop(32, padding = 3),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = means, 
                                                    std = stds)
                           ])

    test_transforms = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean = means, 
                                                    std = stds)
                           ])

    train_data = datasets.CIFAR10('data', 
                                  train = True, 
                                  download = True, 
                                  transform = train_transforms)

    test_data = datasets.CIFAR10('data', 
                                 train = False, 
                                 download = True, 
                                 transform = test_transforms)

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
    model = cnv()
    if args.cuda:
        torch.cuda.set_device(0)
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    # test model
    if args.eval:
        model = torch.load(save_path)
        test()
    # test on single image
    if args.test_on_image:
        print("[DEBUG] loading model and opening image . . .")
        model = torch.load(save_path)
        image_tensor = np.fromfile(args.test_on_image,dtype=np.uint8)
        image_tensor=image_tensor[1:image_tensor.size]
        image = np.zeros((128,3,32, 32),np.uint8)

        #Convert from planar rgb to tensor 128 times
        for b in range(128):
          for ch in range(3):
            k = 0
            j = 0
            for i in range(32*32):
              image[b,ch,k,j] = image_tensor[i+32*32*ch]
              j = j + 1
              if(j == 32):
                k = k+1
                j = 0
        image_tensor = torch.Tensor(image)
        image_tensor.type('torch.FloatTensor') 
        image_tensor = image_tensor.to(device='cuda:0')
        
        print("Eval result: {}".format(model.forward(image_tensor)[0]))
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
            #Learning rate scaling.
            if epoch%40==0 and epoch != 0:
                new_lr = optimizer.param_groups[0]['lr']*0.1
                print("[INFO] Scaling lr to {}.".format(new_lr))
                optimizer.param_groups[0]['lr']=new_lr

        #Finished Training. Load the best model and test it.
        model = torch.load(save_path)        
        test()
