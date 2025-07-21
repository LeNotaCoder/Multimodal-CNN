import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, compressed_channels):
        super(MultiScaleAttention, self).__init__()
        self.compress_fn = nn.Conv2d(in_channels, compressed_channels, kernel_size=1)
        self.compress_an = nn.Conv2d(in_channels, compressed_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(compressed_channels, in_channels, kernel_size=1)

    def forward(self, An_prev, Fn):
        B, C, H, W = Fn.shape

        Fc_n = self.compress_fn(Fn)            
        Ac_prev = self.compress_an(An_prev)     

        Fc_n_flat = Fc_n.view(B, -1, H * W)      
        Ac_prev_flat = Ac_prev.view(B, -1, H * W)  

        attn_map = torch.bmm(Fc_n_flat, Ac_prev_flat.transpose(1, 2))  
        attn_map = F.softmax(attn_map, dim=-1)

        attended = torch.bmm(attn_map, Fc_n_flat.transpose(1, 2))  
        attended = attended.transpose(1, 2).view(B, -1, H, W)  

        output = attended + Fc_n
        output = self.output_proj(output)

        return output


class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=1):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.scale = (in_channels // heads) ** 0.5

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()

        Q = self.query(x).view(B, self.heads, C // self.heads, H * W)   
        K = self.key(x).view(B, self.heads, C // self.heads, H * W)     
        V = self.value(x).view(B, self.heads, C // self.heads, H * W)   

        attn_weights = torch.einsum('bhcn,bhcm->bhnm', Q, K) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.einsum('bhnm,bhcm->bhcn', attn_weights, V)
        attn_output = attn_output.contiguous().view(B, C, H, W) 

        out = attn_output + self.value(x)
        return self.output_proj(out)


class AttentionCNN(nn.Module):
    def __init__(self):
        super(AttentionCNN, self).__init__()

        # Assume input channels for each image is 1
        in_channels = 64
        self.SA = SelfAttention(in_channels=in_channels)
        self.MSA1 = MultiScaleAttention(in_channels=in_channels, compressed_channels=16)
        self.MSA2 = MultiScaleAttention(in_channels=in_channels, compressed_channels=16)
        self.MSA3 = MultiScaleAttention(in_channels=in_channels, compressed_channels=16)

        self.conv1 = nn.Conv2d(1, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 224 * 224, 4096),
            nn.ReLU(),
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def features(self, x):
        # x shape: [B, 4, 224, 224] -> split into 4 images
        x_0 = x[:, 0:1, :, :]  # [B, 1, H, W]
        x_1 = x[:, 1:2, :, :]
        x_2 = x[:, 2:3, :, :]
        x_3 = x[:, 3:4, :, :]

        A_1 = self.SA(self.conv1(x_0))
        F_2 = self.conv1(x_1)
        A_2 = self.MSA1(A_1, F_2)

        F_3 = self.conv2(x_2)
        A_3 = self.MSA2(A_2, F_3)

        F_4 = self.conv3(x_3)
        A_4 = self.MSA3(A_3, F_4)

        output = self.conv_final(A_4)
        return output
    
    def forward(self, x):
        features = self.features(x)
        output = self.fc_layers(features)
        return output


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
    
