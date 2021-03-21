import torch
import numpy as np
from torch.nn import Parameter
import torch.nn as nn


class layer(torch.nn.Module):

    def __init__(self):
        super(layer, self).__init__()
        n = 6
        self.w = torch.nn.Parameter(torch.rand((n,1,128,128),dtype=torch.cfloat))
        self.b = torch.nn.Parameter(torch.rand((n,128,128),dtype=torch.cfloat))
        torch.nn.init.normal_(self.w)
        torch.nn.init.normal_(self.b)

    def forward(self,inputs):
        x = inputs
        #傅里叶1
        x = torch.fft.fft2(x*self.w[0] + self.b[0])
        #傅里叶2
        x = torch.fft.fft2(x*self.w[1] + self.b[1])
        #逆傅里叶1
        x = torch.fft.ifft2(x*self.w[2] + self.b[2])
        #角度
        x = x.angle()
        #输入平面只保留相位的相息图
        x = torch.cos(x)+1j*torch.sin(x) 
        #傅里叶3
        x = torch.fft.fft2(x*self.w[3] + self.b[3])
        #逆傅里叶2
        x = torch.fft.ifft2(x*self.w[4] + self.b[4])
        #逆傅里叶3
        x = torch.fft.ifft2(x*self.w[5] + self.b[5])
        #角度
        x = x.angle();
        x = inputs.abs()*(torch.cos(x)+1j*torch.sin(x));
        return x


class Net(torch.nn.Module):
    """
    phase only modulation
    """
    def __init__(self, size, num_layers=5):

        super(Net, self).__init__()
        
        self.block = nn.ModuleList([layer() for i in range(10)])

    def forward(self, inputs):
        x = inputs
        for i,layer in enumerate(self.block):
            x = layer(x)
        x = (x.angle()+3.14)/2.68;
        return x


if __name__ == '__main__':
    print(Net())
