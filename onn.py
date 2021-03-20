import torch
import numpy as np
from torch.nn import Parameter




class DiffractiveLayer(torch.nn.Module):

    def __init__(self,size):
        super(DiffractiveLayer, self).__init__()
        self.size = size                         # SIZE * SIZE neurons in one layer
        self.distance = 0.03                    # distance bewteen two layers (3cm)
        self.ll = 0.08                          # layer length (8cm)
        self.wl = 3e8 / 0.4e12                 # wave length

        #self.distance = Parameter(torch.tensor(0.03))                    # distance bewteen two layers (3cm)
        #self.ll = Parameter(torch.tensor(0.08))                          # layer length (8cm)
        #self.wl = Parameter(torch.tensor(3e8 / 0.4e12))                 # wave length
        self.fi = 1 / self.ll                   # frequency interval
        self.wn = 2 * 3.1415926 / self.wl       # wave number




        # self.phi
        self.phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size), dtype=np.complex64)
        # h
        h = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * self.phi))
        # self.h
        self.h = torch.nn.Parameter(torch.stack((torch.from_numpy(h.real), torch.from_numpy(h.imag)), dim=-1), requires_grad=True)
        self.act = torch.nn.ReLU()

    def forward(self, waves):
        # waves
        temp = torch.fft(waves, signal_ndim=2)
        k_space_real = self.h[..., 0] * temp[..., 0] - self.h[..., 1] * temp[..., 1]
        k_space_imag = self.h[..., 0] * temp[..., 1] + self.h[..., 1] * temp[..., 0]

        k_space_real = self.act(k_space_real)
        k_space_imag = self.act(k_space_imag)

        k_space = torch.stack((k_space_real, k_space_imag), dim=-1)
        # angular_spectrum
        angular_spectrum = torch.ifft(k_space, signal_ndim=2)
        angular_spectrum += waves
        return angular_spectrum


class Net(torch.nn.Module):
    """
    phase only modulation
    """
    def __init__(self, size, num_layers=5):

        super(Net, self).__init__()
        # self.phase 
        self.phase = [torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=(size, size)).astype('float32'))) for _ in range(num_layers)]
        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), self.phase[i])
        self.diffractive_layers = torch.nn.ModuleList([DiffractiveLayer(size) for _ in range(num_layers)])
        self.last_diffractive_layer = DiffractiveLayer(size)

        self.conv = torch.nn.Conv2d(1,1,3,stride=2,padding=1)
        self.act = torch.nn.ReLU()


    def forward(self, inputs):
        # x 
        x = inputs
        for index, layer in enumerate(self.diffractive_layers):

            temp = layer(x)
            exp_j_phase = torch.stack((torch.cos(self.phase[index]), torch.sin(self.phase[index])), dim=-1)
            x_real = temp[..., 0] * exp_j_phase[..., 0] - temp[..., 1] * exp_j_phase[..., 1]
            x_imag = temp[..., 0] * exp_j_phase[..., 1] + temp[..., 1] * exp_j_phase[..., 0]
            x = torch.stack((x_real, x_imag), dim=-1)
            #x = self.relu(x)
        x = self.last_diffractive_layer(x)
        #x = self.sigmoid(x)
        # x_abs 
        x_abs = torch.sqrt(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1])
        x_abs = x_abs.unsqueeze(dim=1)
        x_abs = self.conv(x_abs)
        return x_abs


if __name__ == '__main__':
    print(Net())
