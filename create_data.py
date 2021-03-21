import scipy
from scipy.io import loadmat
import numpy
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path

mse = torch.nn.MSELoss()

N = 128

mask1 = loadmat('mask1.mat')['mask1']

def generate(runs,mask,inpath):
    cmask1 = numpy.conj(mask)
    XRGB = Image.open(inpath)
    X0 = XRGB.convert('L')

    X = numpy.array(X0,dtype=numpy.float64)
    h=0.532e-3      #波长(mm), 可按需要修改
    w=h
    pi = numpy.pi
    k=2*pi/h
    pix=0.0064      #SLM像素宽度(mm), 可按需要修改
    L=N*pix         #SLM宽度(mm)
    z0=250           #衍射距离(mm),
    L0=h*z0/pix     #重建像平面宽度(mm)
    Lf=1/pix        #频域宽度

    u,v=torch.meshgrid(torch.linspace(-Lf/2,Lf/2,steps=N),torch.linspace(-Lf/2,Lf/2,steps=N))
    u = u.T
    v = v.T


    H=numpy.exp(1j*k*z0*numpy.sqrt(1-(w*u)**2-(w*v)**2))
    CH=torch.conj(H);

    Y=torch.tensor(X).to(torch.double)

    U0=Y;

    # U0是inputs

    for p in range(1,runs):
        #菲涅耳衍射的S-FFT计算开始
        n=torch.arange(1,N+1)
        x=-L0/2+L0/N*(n-1)			
        y=x
        yy,xx = torch.meshgrid(y,x); 
        yy = yy.T
        xx = xx.T
        Fresnel=numpy.exp(-1j*k/2/z0*(xx**2+yy**2)) #负号表示逆衍射
        f2=U0*Fresnel;
        # 傅里叶1
        Uf=torch.fft.fft2(f2);
        x=-L/2+L/N*(n-1) #SLM宽度取样(mm) 					
        y=x;
        [yy,xx] = torch.meshgrid(y,x); 
        yy = yy.T
        xx = xx.T
        phase=numpy.exp(-1j*k*z0)/(-1j*h*z0)*numpy.exp(-1j*k/2/z0*(xx**2+yy**2));
        Uf=Uf*phase*cmask1;
        #接下来用角谱法计算从目前的平面逆衍射到输入平面相位板的过程
        #傅里叶2，逆傅里叶1
        Ui=torch.fft.ifft2(torch.fft.fft2(Uf)*CH) #计算输入平面光波复振幅

        #菲涅耳衍射的S-FFT计算结束
        #计算角度
        Phase=Ui.angle()+pi
        Ih=numpy.uint8(Phase/2/pi*255) #形成0-255灰度级的相息图 最终输出
        Uii=torch.cos(Phase-pi)+1j*torch.sin(Phase-pi) #输入平面只保留相位的相息图
        Um=torch.fft.ifft2(torch.fft.fft2(Uii)*H)*mask1 #中间随机相位板后表面的复振幅 
        n=torch.arange(1,N+1)
        x=-L/2+L/N*(n-1)
        y=x;
        [yy,xx] = torch.meshgrid(y,x);
        yy = yy.T
        xx = xx.T
        Fresnel=numpy.exp(1j*k/2/z0*(xx**2+yy**2))
        f2=Um*Fresnel
        Uf=torch.fft.ifft2(f2);
        x=-L0/2+L0/N*(n-1) #重建像平面宽度取样(mm) 					
        y=x;
        [yy,xx] = torch.meshgrid(y,x); 
        yy = yy.T
        xx = xx.T
        phase=numpy.exp(1j*k*z0)/(1j*h*z0)*numpy.exp(1j*k/2/z0*(xx**2+yy**2));
        Uf=Uf*phase;

        #保持相位不变，引用原图振幅，重新开始新一轮计算
        Phase=Uf.angle();
        U0=Y.abs()*(torch.cos(Phase)+1j*torch.sin(Phase));
    img = Image.fromarray(Ih)
    return img,Ih

def generate_batch():

    for i,path in enumerate(Path('./data/out2/').glob('*.jpg')):
        print(str(path))
        img = generate(mask1, str(path))
        img.save('data/myout/%d.jpg'%i)

if __name__ == '__main__':
    runs = 15
    mse = torch.nn.MSELoss()
    img,Ih = generate(runs, mask1,'./data/Train128///1.bmp')
    img.save('test.png')
    img2 = Image.open('./data/LTrain128/1.bmp').convert('L')
    img2.save('test2.png')
    Ih2 = numpy.array(img2)
    print(mse(torch.tensor(Ih,dtype=torch.float64),torch.tensor(Ih2,dtype=torch.float64)))

