import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision

from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchsummary import summary

#使用GPU進行運算
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

#圖片路徑
PATH_train = r"C:\Users\ucl\Desktop\AIdeal_competition\data_aug_jack\train" 
PATH_val = r"C:\Users\ucl\Desktop\AIdeal_competition\data_aug_jack\val"

TRAIN = Path(PATH_train)
VALID = Path(PATH_val)

print(TRAIN)
print(VALID)

#參數
num_workers = 0
learning_rate = 0.0001
EPOCH = 100
batch_size = 30

#resize和正歸化
train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

#加載圖像資料夾
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
valid_data = datasets.ImageFolder(VALID, transform=valid_transforms)

#數據集上的迭代
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,  num_workers=num_workers, shuffle=True)
#迭代對象
images, labels = next(iter(train_loader))
images.shape, labels.shape
#==========================================================================================================================================================
#驗證筆數
val_num = len(valid_loader.dataset)
# 設定 GPU
device = torch.device("cuda")

#使用遷移學習(resnet101)
model = torchvision.models.resnet101(pretrained=True, progress=True)
#由於resnet101的fc有1000個，而我們可以固定的比例分割
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 6)

#查看模型架構
summary(model.cuda(),(3, 224, 224))

#遷移學習->freeze(copy 四層)
for name, parameter in model.named_parameters():
    if name == 'layer4.0.conv1.weight':
        break
    parameter.requires_grad = False

#在GPU上運算
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.to(device)
#訓練次數
n_epochs = EPOCH
valid_loss_min = np.Inf  

#定義損失函數&優化器
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0)

#學習率下降(更新學習率)，原因使優化器快速收斂
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
train_losses, val_losses = [], []

##############################################################################################
#train model
for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    valid_loss = 0.0
    print('\nrunning epoch: {}'.format(epoch))

    model.train()
    with tqdm(train_loader) as pbar:
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.to(device), target.to(device)
            #優化器初始梯度為0
            optimizer.zero_grad()
            output = model(data)
             #計算損失函數
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
             #更新loss
            train_loss += loss.item()*data.size(0)
            #訓練條
            pbar.update(1)
            pbar.set_description('train')
            pbar.set_postfix(
                **{
                    'epochs': str('{}/{}'.format(epoch, n_epochs)),
                    'loss': loss.item(),
                    'lr': optimizer.state_dict()['param_groups'][0]['lr']
                })
        scheduler.step()
    #############################################################################################
    #val model
    model.eval()
    val_accuracy = 0
    for data, target in tqdm(valid_loader):
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        predict_y = torch.max(output, dim=1)[1]
        val_accuracy = val_accuracy + \
            (predict_y == target.to(device)).sum().item()

    accuracy = val_accuracy / val_num
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(valid_loss)

    #print出訓練的loss和驗證的loss&準確率
    print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
    print('validation accuracy = ', accuracy)

    #如果驗證有持續減少的話save model
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        # torch.save(model.state_dict(), 'C:\\Users\\ucl\\Desktop\\AIdeal_competition\\model\\my_model_resnet101.pth')
        valid_loss_min = valid_loss

    torch.save(model.state_dict(), 'C:/Users/ucl/Desktop/AIdeal_competition/model/epoch%d-train_loss%.4f-val_loss%.4f-validation accuracy%.4f.pth' %(epoch, train_loss, valid_loss, accuracy))

# 繪製圖
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_losses, label='train_losses')
plt.plot(val_losses, label='val_losses')
plt.legend(loc='best')
plt.savefig('C:\\Users\\ucl\\Desktop\\AIdeal_competition\\figure\\Loss.jpg')
plt.show()
