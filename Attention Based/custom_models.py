import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()

        self.conv_similarity = nn.Conv2d(1, 1, kernel_size=3, stride=5, padding=2)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.fc_layers = nn.Sequential(
            nn.Linear(32768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def features(self, x):
        # x shape: [B, 4, H, W]
        x_0 = x[:, 0:1, :, :]
        x_1 = x[:, 1:2, :, :]
        x_2 = x[:, 2:3, :, :]
        x_3 = x[:, 3:4, :, :]

        s0 = self.conv_similarity(x_0)
        s1 = self.conv_similarity(x_1)
        s2 = self.conv_similarity(x_2)
        s3 = self.conv_similarity(x_3)

        attention = (s0 + s1 + s2 + s3) / 4  # shape: [B, 1, H', W']
        attention = F.interpolate(attention, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = x * attention  # broadcasting over 4 channels

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.adaptive_avg_pool2d(x, (8, 8))
        
        return x    
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_layers(x)
        return x

    

class FineFeatureCNN(nn.Module):
    def __init__(self):
        super(FineFeatureCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def extract_features(self, x):
        #Extract features without classification
        return self.features(x)


class oct_fine_feat(nn.Module):
    def __init__(self):
        super(oct_fine_feat, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        print("oct_fine_feat started")
        x = self.features(x)
        x = self.classifier(x)
        return x

class fundus_vgg16(nn.Module):
    def __init__(self):
        super(fundus_vgg16, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FinalDualModel(nn.Module):
    def __init__(self, oct_m, fun):
        super(FinalDualModel, self).__init__()
        
        self.oct_m = oct_m
        self.fun = fun
        
        # Freeze feature extraction layers
        for param in self.oct_m.parameters():
            param.requires_grad = True
        for param in self.fun.parameters():
            param.requires_grad = True
        
        self.fc_layers = nn.Sequential(
            nn.Linear(8192, 4096),  # Combined features
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
        )
        
    def forward(self, x1, x2):
        feat1 = self.oct_m.features(x1)
        feat2 = self.fun.features(x2)
        feat1 = feat1.view(feat1.size(0), -1)
        feat2 = feat2.view(feat2.size(0), -1)
        combined = torch.cat((feat1, feat2), dim=1)
        output = self.fc_layers(combined)
        return output
    

    
    
class Dualbranch(nn.Module):
    def __init__(self): 
        super(Dualbranch, self).__init__()
        
        self.branch1 = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((112,112)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((56,56)),
            
            )
        
        self.branch2 = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            )
        
        self.combined = nn.Sequential(
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            
            )
        
        self.fc_layers = nn.Sequential(
            
            nn.Flatten(),
            nn.Linear(100352, 4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            
            )
        
    def forward(self, x):
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        features = torch.concat((b1, b2), dim=1)
        
        
        combined = self.combined(features)
        
        output = self.fc_layers(combined)
        
        return output
    
    def features(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        features = torch.concat((b1, b2), dim=1)
        combined = self.combined(features)
        
        return combined
    
