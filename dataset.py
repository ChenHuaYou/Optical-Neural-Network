import numpy as np
import math
from PIL import Image
import torch
from pathlib import Path
import random
import torchvision as tv




class MyDataset(torch.utils.data.Dataset):

    def __init__(self, x_data, label_dir):
        self.x = x_data
        self.y = [label_dir+path.name for path in self.x]
        self.image_transformer = tv.transforms.Compose([
            #tv.transforms.Resize((28,28)),
            tv.transforms.Grayscale(),
            tv.transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = Image.open(self.x[idx])
        x = self.image_transformer(x)
        y = Image.open(self.y[idx])
        y = self.image_transformer(y)

        return x,y

if __name__ == '__main__':

    split = 0.8
    x_data = list(Path('data/Train128/').glob('*.bmp'))
    totalNum = len(x_data)

    train_x_data = x_data[:int(totalNum*split)]
    test_x_data = x_data[int(totalNum*split):]

    train_data = MyDataset(train_x_data,'data/LTrain128/')
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=4,shuffle=True)
    for i,batch in enumerate(train_loader):
        inputs, outputs = batch
        img = Image.fromarray(inputs[0].reshape((128,128)))
        img.save('test.png')
        exit(0)

