# 自動光學檢測(pytorch)-------AIdea競賽

## 分別使用rest101和resnet152進行訓練
先將圖片resize成224*224的大小
使用遷移學習的方式載入resnet101的預訓練模型，並對前四層進行冷凍，進行訓練

其參數如下<br>
num_workers = 0 <br>
learning_rate = 0.0001 <br>
EPOCH = 100 <br>
batch_size = 30 <br>
優化器使用adam <br>

## test
依據驗證資料集將圖片分類在csv中，依據實驗結果resnet152的驗證結果較佳，在比賽中有著27/98<br>
![image](https://github.com/MING-SIAN/Automated-Optical-Inspection/blob/main/%E7%B5%90%E6%9E%9C/%E5%90%8D%E6%AC%A1.jpg)

