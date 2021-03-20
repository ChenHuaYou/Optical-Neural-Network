from PIL import Image
import numpy as np
import torch


mse = torch.nn.MSELoss()

img1 = Image.open('test_1.png').convert('L')
img1 = np.array(img1)
img1 = torch.tensor(img1,dtype=torch.float64).unsqueeze(dim=0)

img2 = Image.open('data/LTrain128/1.bmp').convert('L')
img2 = np.array(img2)
img2 = torch.tensor(img2,dtype=torch.float64).unsqueeze(dim=0)


res = mse(img1,img2)
print(res)
