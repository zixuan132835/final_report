import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2

# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc

# 我们通过继承Dataset类来创建我们自己的数据加载类，命名为FaceDataset
class FaceDataset(data.Dataset):
  
    # 初始化
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        df_path = pd.read_csv(root + '\\image_emotion.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\image_emotion.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

   

    # 读取某幅图片，item为索引号
    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])
        # 读取单通道灰度图
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 高斯模糊
        # face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)
        # 直方图均衡化
        face_hist = cv2.equalizeHist(face_gray)
        # 像素值标准化
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0 # 为与pytorch中卷积神经网络API的设计相适配，需reshape原图
        # 用于训练的数据需为tensor类型
        face_tensor = torch.from_numpy(face_normalized) # 将python中的numpy数据类型转化为pytorch中的tensor数据类型
        face_tensor = face_tensor.type('torch.FloatTensor') # 指定为'torch.FloatTensor'型，否则送进模型后会因数据类型不匹配而报错
        label = self.label[item]
        return face_tensor, label




    # 获取数据集样本个数
    def __len__(self):
        return self.path.shape[0]


class FaceCNN(nn.Module):
    # 初始化网络结构
    def __init__(self):
        super(FaceCNN, self).__init__()

        # 第一次卷积、池化
        self.conv1 = nn.Sequential(
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 1, 48, 48), output:(bitch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # 卷积层
            nn.BatchNorm2d(num_features=64), # 归一化
            nn.RReLU(inplace=True), # 激活函数
            # output(bitch_size, 64, 24, 24)
            nn.MaxPool2d(kernel_size=2, stride=2), # 最大值池化
        )

        # 第二次卷积、池化
        self.conv2 = nn.Sequential(
            # input:(bitch_size, 64, 24, 24), output:(bitch_size, 128, 24, 24), (24-3+2*1)/1+1 = 24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 128, 12 ,12)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积、池化
        self.conv3 = nn.Sequential(
            # input:(bitch_size, 128, 12, 12), output:(bitch_size, 256, 12, 12), (12-3+2*1)/1+1 = 12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            # output:(bitch_size, 256, 6 ,6)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 参数初始化
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 数据扁平化
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y

def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    model = FaceCNN()
    # 损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # 学习率衰减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        # scheduler.step() # 学习率衰减
        model.train() # 模型训练
        for images, emotion in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model.forward(images)
            # 误差计算
            loss_rate = loss_function(output, emotion)
            # 误差的反向传播
            loss_rate.backward()
            # 更新参数
            optimizer.step()

        # 打印每轮的损失
        print('After {} epochs , the loss_rate is : '.format(epoch+1), loss_rate.item())
        if epoch % 5 == 0:
            model.eval() # 模型评估
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('After {} epochs , the acc_train is : '.format(epoch+1), acc_train)
            print('After {} epochs , the acc_val is : '.format(epoch+1), acc_val)

    return model

def main():
    # 数据集实例化(创建数据集)
    train_dataset = FaceDataset(root='face_images/train_set')
    val_dataset = FaceDataset(root='face_images/verify_set')
    # 超参数可自行指定  epochs = 100 太大了 我跑了一个晚上第二天早上还没结束，但是参数越高训练精度越准确，有GPU的试一下
    model = train(train_dataset, val_dataset, batch_size=128, epochs=50, learning_rate=0.1, wt_decay=0)
    # 保存模型
    torch.save(model, 'model/model_cnn.pkl')


if __name__ == '__main__':
    main()

