import csv
import cv2
import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import pandas as pd

from PIL import Image
from model import *
from torchvision import datasets, models, transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset

#TestData_loader
class TestDataset(Dataset):
    def __init__(self, img_list, transform):
        super(TestDataset, self).__init__()
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        trans_img = self.transform(img)
        return trans_img


#使用GPU進行運算
device = torch.device("cuda")

#路徑
image_path = r"C:\Users\ucl\Desktop\AIdeal_competition\data_aug_jack\test"
classes = os.listdir(r'C:\Users\ucl\Desktop\AIdeal_competition\data_aug_jack\train')

#resize和正歸化
test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#model
model = torchvision.models.resnet101(pretrained=True, progress=True)
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 6)

#載入預訓練權重
model.load_state_dict(torch.load(r'C:\Users\ucl\Desktop\AIdeal_competition\train_resnet152\epoch63-train_loss0.0002-val_loss0.0217-validation accuracy0.9980.pth'))

model.to(device)
model.eval()

result = []

#csv路徑
df = pd.read_csv(r"C:\Users\ucl\Desktop\AIdeal_competition\upload_sample.csv")
img_list = []
#載入範例得知test資料集名稱
for index, row in tqdm(df.iterrows()):
    img = os.path.join(image_path, row['ID'])
    img_list.append(img)

test_data = TestDataset(img_list, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=20)



results = []                                     
model.eval()         
with torch.no_grad():                                 
    for data in tqdm(test_loader):
        data = data.to(device)
        pred = model(data)
        predict_y = torch.max(pred, dim=1)[1]
        results = np.hstack((results, np.array(predict_y.cpu().detach())))

df['Label'] = results

df.to_csv('C:\\Users\\ucl\\Desktop\\AIdeal_competition\\train_resnet152\\csv\\epoch63-train_loss0.0002-val_loss0.0217-validation accuracy0.9980.csv', index=None)


