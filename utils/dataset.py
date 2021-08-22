import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    # 為此class的初始化function，根據指定的圖片路徑，存放到self.imgs_path列表中。
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    # 此function是data獲取function，在這個函數裡，可以寫data如何讀取，如何處理，並可以做data的
    # 預處理，數據增強。
    # 這邊處理方式簡單，只將圖片讀取，並處理成單通道圖片，同時，因為label的圖片像素點是0和255，因此
    # 將其改為0和1，同時隨機進行數據增強。
    # 這裡的增強方式(function augment)，僅做了簡單的旋轉。
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        # 回傳data pair
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)