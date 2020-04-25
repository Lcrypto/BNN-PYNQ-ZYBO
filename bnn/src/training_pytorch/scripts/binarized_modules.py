import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np

class QuantizeAct(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, numbits):
        ctx.save_for_backward(input)
        if numbits == 1:
            return input.sign()
        elif numbits == 2:
            return torch.floor(input + 0.5)
        elif numbits == 8:
            # not sure if clamping is needed here as tensor already passed through HardTanH (?)
            # min and max values calculated using 2.6 format. i.e. minval = -(2**intbits-1)
            # max val = -(minval) - 2**-fracbits
            # so 8 bit config only uses this format.
            cliped = torch.clamp(input, -2, 1.984375)
            cliped = torch.round(cliped * 2.**6)/2.**6
            return cliped
        else:
            return torch.floor(input.add(1).div(2).clamp_(0, 0.999).mul(2**numbits-1)).sub((2**numbits-1)//2)#.div((2**numbits-1)//2)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

def QuantizeWeights(tensor, numbits):
    if numbits == 1:
        return tensor.sign()
    elif numbits == 2:
        return torch.floor(tensor + 0.5)
    elif numbits == 8:
        # maybe clamping is needed at the initialization time (?)
        # During training weights are already being clipped between -1, 1
        cliped = torch.clamp(tensor, -2, 1.984375)
        cliped = torch.round(cliped * 2.**6)/2.**6
        return cliped
    else:
        return torch.floor(tensor.add(1).div(2).clamp_(0, 0.999).mul(2**numbits-1)).sub((2**numbits-1)//2).div((2**numbits-1)//2)
        

class Quantizer(nn.Module):
    def __init__(self, numbits):
        super(Quantizer, self).__init__()
        self.numbits=numbits

    def forward(self, input):
        return QuantizeAct.apply(input, self.numbits)

class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        self.wb = kargs[0]
        kargs = kargs[1:]
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=QuantizeWeights(self.weight.org, self.wb)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out

class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        self.wb = kargs[0]
        kargs = kargs[1:]
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
    
    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=QuantizeWeights(self.weight.org, self.wb)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()
    
    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(0).div(target.numel())
        return loss

    def backward(self,grad_output,retain_graph=True): 
       print('xd','\n')
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output