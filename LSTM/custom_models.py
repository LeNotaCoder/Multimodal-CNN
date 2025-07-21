import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(MultiScaleLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        
        self.input_proj = nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=1)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, An_prev, Fn):
        B, C, H, W = Fn.shape

        concat = torch.cat([Fn, An_prev], dim=1)
        projected = self.input_proj(concat) 

        seq = projected.view(B, self.hidden_dim, -1).permute(0, 2, 1)

        lstm_out, _ = self.lstm(seq) 

        lstm_out = lstm_out.permute(0, 2, 1).contiguous().view(B, self.hidden_dim, H, W)

        out = self.output_proj(lstm_out) 

        return out

class CNNWithLSTMBlocks(nn.Module):
    def __init__(self, input_channels=4, num_classes=2):
        super(CNNWithLSTMBlocks, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm1 = MultiScaleLSTM(32, hidden_dim=64)
        self.lstm2 = MultiScaleLSTM(32, hidden_dim=64)
        self.lstm3 = MultiScaleLSTM(32, hidden_dim=64)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def features(self, x):

        x_0 = x[:, 0:1, :, :]
        x_1 = x[:, 1:2, :, :]
        x_2 = x[:, 2:3, :, :]
        x_3 = x[:, 3:4, :, :]

        F_0 = self.encoder(x_0)
        F_1 = self.encoder(x_1)
        F_2 = self.encoder(x_2)
        F_3 = self.encoder(x_3)
        
        A_1 = self.lstm1(F_0, F_1)
        A_2 = self.lstm2(A_1, F_2)
        A_3 = self.lstm3(A_2, F_3)

        return A_3 
    
    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out


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
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
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
    
