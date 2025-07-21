import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

sys_path = "/Users/yadyneshsonale/k/"
import sys
sys.path.append(f'{sys_path}functions.py')
from functions import apply_algo, apply_fun_pre


class OCTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
    
    
class PairedDataSet(Dataset):
    def __init__(self, fundus_data, oct_data, labels, fun, model, transform=None, transform2=None, device='cpu'):
        self.fundus_data = fundus_data
        self.oct_data = oct_data
        self.labels = labels
        self.transform = transform
        self.transform2 = transform2
        self.model = model.to(device)
        self.fun = fun.to(device)
        self.device = device
        self.model.eval()
        self.fun.eval()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        fundus_image = self.fundus_data[idx]
        oct_image = self.oct_data[idx]
        label = self.labels[idx]
        
        if not isinstance(fundus_image, np.ndarray):
            fundus_image = np.array(fundus_image)
        if not isinstance(oct_image, np.ndarray):
            oct_image = np.array(oct_image)
        
        # Convert and normalize OCT image
        oct_image_tensor = torch.from_numpy(oct_image).permute(2, 0, 1).float()  # (3, H, W)
        if self.transform2:
            oct_image_tensor = self.transform2(oct_image_tensor)
        oct_image_tensor = oct_image_tensor.unsqueeze(0).to(self.device)

        # Extract features from model
        with torch.no_grad():
            feature = self.model.features(oct_image_tensor)  # (1, 64, 8, 8)
        oct_feature = feature.squeeze(0).cpu()  # (64, 8, 8)

        # Convert and normalize Fundus image
        fundus_image_tensor = torch.from_numpy(fundus_image).permute(2, 0, 1).float()
        if self.transform:
            fundus_image_tensor = self.transform(fundus_image_tensor)
            
        
        fundus_image_tensor = fundus_image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            fun_feature = self.fun.features(fundus_image_tensor)  # (1, 64, 8, 8)
        fundus_feature = fun_feature.squeeze(0).cpu()  # (64, 8, 8)
        

        return fundus_feature, oct_feature, label
    
    
    


def get_loaders(stri, batch_size=32):
    dataset_path = "/home/cs23b1055/" + stri
    
    print("get_loaders started")
    dr, normal = [], []
    n = 3001

    for i in range(1, n):
        dr_feata = f"{dataset_path}/DR/{i}_a.jpg"
        dr_featb = f"{dataset_path}/DR/{i}_b.jpg"
        dr_featc = f"{dataset_path}/DR/{i}_c.jpg"
        dr_featd = f"{dataset_path}/DR/{i}_d.jpg"
        
        dr_feat1 = cv2.imread(dr_feata, cv2.IMREAD_GRAYSCALE)
        dr_feat2 = cv2.imread(dr_featb, cv2.IMREAD_GRAYSCALE)
        dr_feat3 = cv2.imread(dr_featc, cv2.IMREAD_GRAYSCALE)
        dr_feat4 = cv2.imread(dr_featd, cv2.IMREAD_GRAYSCALE)
        
        dr_feat = np.stack([dr_feat1, dr_feat2, dr_feat3, dr_feat4], axis=-1)


        dr.append(dr_feat)
        
        n_feata = f"{dataset_path}/NORMAL/{i}_a.jpg"
        n_featb = f"{dataset_path}/NORMAL/{i}_b.jpg"
        n_featc = f"{dataset_path}/NORMAL/{i}_c.jpg"
        n_featd = f"{dataset_path}/NORMAL/{i}_d.jpg"

        n_feat1 = cv2.imread(n_feata, cv2.IMREAD_GRAYSCALE)
        n_feat2 = cv2.imread(n_featb, cv2.IMREAD_GRAYSCALE)
        n_feat3 = cv2.imread(n_featc, cv2.IMREAD_GRAYSCALE)
        n_feat4 = cv2.imread(n_featd, cv2.IMREAD_GRAYSCALE)
        
        n_feat = np.stack([n_feat1, n_feat2, n_feat3, n_feat4], axis=-1)
        
        normal.append(n_feat)

    X = np.array(dr + normal, dtype=np.uint8)
    y = np.array([1]*len(dr) + [0]*len(normal))
    
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    
    print(len(X), len(y))

    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]),
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    train_dataset = OCTDataset(X_train, y_train, transform=train_transform)
    test_dataset = OCTDataset(X_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader



def get_loaders_fun(stri, batch_size=32):
    dataset_path = "/home/cs23b1055/" + stri
    print("get_loaders_fun started")
    dr, normal = [], []
    n = 1409

    for i in range(1, n):
        dr_path = os.path.join(dataset_path, f"DR/{i}.jpg")
        n_path = os.path.join(dataset_path, f"NORMAL/{i}.jpg")

        dr_image = apply_fun_pre(dr_path)
        n_image = apply_fun_pre(n_path)

        dr.append(dr_image)
        normal.append(n_image)

        

    X = np.array(dr + normal, dtype=np.uint8)
    y = np.array([1]*len(dr) + [0]*len(normal))

    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    train_dataset = OCTDataset(X_train, y_train, transform=train_transform)
    test_dataset = OCTDataset(X_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader




