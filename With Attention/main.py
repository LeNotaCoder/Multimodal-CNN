import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

sys_path = "/home/cs23b1055/"

from preprocessing import apply_fun_pre, get_oct_image
from dataloaders import PairedDataSet
from custom_models import AttentionCNN, fundus_vgg16, oct_fine_feat, FinalDualModel, Dualbranch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AttentionCNN()
model.load_state_dict(torch.load(f"{sys_path}attention_oct_model.pth"))  # load weights
model = model.to(device) 

model2 = Dualbranch()
model2.load_state_dict(torch.load(f"{sys_path}fun_model.pth"))  # load weights
model2 = model2.to(device)

# Data loading
path1 = "/home/cs23b1055/images/eyefundus.csv"
path2 = "/home/cs23b1055/images/oct.csv"

data1 = pd.read_csv(path1)
data2 = pd.read_csv(path2)

fundus_dict = data1.set_index('Name')['DR'].to_dict()
oct_dict = data2.set_index('Name')['DR'].to_dict()

data1 = np.array(data1)
data2 = np.array(data2)

d = dict()

for data in data2:
    for datas in data1:
        if data[0][0:7] == datas[0][0:7]:
            if data[0] not in d.keys():
                d[data[0]] = [datas[0]]

keysl = []
for key in d.keys():
    keysl.append(key[0:7])

key = Counter(keysl)

for k in key.keys():
    while key[k] > 1:
        for dkey in d.keys():
            if dkey[0:7] == k:
                del d[dkey]
                break
        key[k] -= 1

fundus_images = []
oct_images = []
fundus_labels = []
oct_labels = []

hh = 0

for key in d.keys():
    fundus_path = "/home/cs23b1055/images/fundus/" + str(d[key][0]) + ".jpg"
    fundus_image = apply_fun_pre(fundus_path) 

    oct_path = "/home/cs23b1055/images/oct/" + str(key) + ".jpg"
    oct_image = get_oct_image(str(key))
    
    hh+=1
    print(hh)


    if fundus_dict[d[key][0]] == '0':
        fundus_label = 0
    elif fundus_dict[d[key][0]] in ['NPDR', 'PDR']:
        fundus_label = 1
    else:
        continue

    oct_label = 1 if oct_dict[key] != '0' else 0

    fundus_images.append(fundus_image / 255.0)
    oct_images.append(oct_image)
    fundus_labels.append(fundus_label)
    oct_labels.append(oct_label)

l = [357, 357, 359, 359]
for i in sorted(l, reverse=True):
    if i < len(fundus_images):
        fundus_images.pop(i)
        fundus_labels.pop(i)
        oct_images.pop(i)
        oct_labels.pop(i)

# Physical oversampling to address class imbalance and increase both classes
labels = np.array(fundus_labels)
class_counts = Counter(labels)
print(f"Class distribution before oversampling: {class_counts}")

# Set target count to 1.5x the majority class count
majority_count = max(class_counts.values())
target_count = int(1.5 * majority_count)

# Oversample both classes
oversampled_fundus = []
oversampled_oct = []
oversampled_labels = []

for label in class_counts.keys():
    indices = np.where(labels == label)[0]
    num_samples = len(indices)
    oversample_factor = target_count // num_samples
    remainder = target_count % num_samples
    
    # Add samples according to oversample_factor
    for _ in range(oversample_factor):
        for idx in indices:
            oversampled_fundus.append(fundus_images[idx])
            oversampled_oct.append(oct_images[idx])
            oversampled_labels.append(fundus_labels[idx])
    
    # Add remaining samples randomly
    if remainder > 0:
        random_indices = np.random.choice(indices, remainder, replace=False)
        for idx in random_indices:
            oversampled_fundus.append(fundus_images[idx])
            oversampled_oct.append(oct_images[idx])
            oversampled_labels.append(fundus_labels[idx])

# Verify new class distribution
print(f"Class distribution after oversampling: {Counter(oversampled_labels)}")



fundus_images = [img for img in oversampled_fundus]
oct_images = [img for img in oversampled_oct]
labels = oversampled_labels

batch_size = 32

fundus_train, fundus_test, oct_train, oct_test, y_train, y_test = train_test_split(
    fundus_images, oct_images, labels, train_size=0.8, random_state=42
)

# Ensure data is in list format
fundus_train = [np.array(img) for img in fundus_train]
oct_train = [np.array(img) for img in oct_train]
fundus_test = [np.array(img) for img in fundus_test]
oct_test = [np.array(img) for img in oct_test]

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transform for validation (no augmentation)
val_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

oct_train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
])

# Define transform for validation (no augmentation)
oct_val_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
])



paired_train = PairedDataSet(fundus_train, oct_train, y_train, model2, model, transform=train_transform, transform2=oct_train_transform)
paired_test = PairedDataSet(fundus_test, oct_test, y_test, model2, model, transform=val_transform, transform2=oct_val_transform)

paired_loader = DataLoader(paired_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
paired_val_loader = DataLoader(paired_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


del model
del model2


# Initialize models
model_oct = oct_fine_feat().to(device)
model_fundus = fundus_vgg16().to(device)
dual_model = FinalDualModel(model_oct, model_fundus).to(device)

# Training parameters
num_epochs = 60
accumulation_steps = 2
optimizer_dual = optim.Adam(dual_model.fc_layers.parameters(), lr=0.001)
criterion_dual = nn.CrossEntropyLoss()

# Lists to store metrics
dual_train_losses = []
dual_train_accuracies = []
dual_val_losses = []
dual_val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    dual_model.train()
    running_loss = 0.0 
    correct = 0
    total = 0
    optimizer_dual.zero_grad()
    
    for i, (fundus_batch, oct_batch, labels_batch) in enumerate(paired_loader):
        fundus_batch = fundus_batch.to(device)
        oct_batch = oct_batch.to(device)
        labels_batch = labels_batch.long().to(device)
        outputs = dual_model(fundus_batch, oct_batch)
        loss = criterion_dual(outputs, labels_batch) / accumulation_steps
        loss.backward()
        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer_dual.step()
            optimizer_dual.zero_grad()
        
    if len(paired_loader) % accumulation_steps != 0:
        optimizer_dual.step()
        optimizer_dual.zero_grad()
    
    train_loss = running_loss / len(paired_loader)
    train_acc = 100 * correct / total
    dual_train_losses.append(train_loss)
    dual_train_accuracies.append(train_acc)

    # Validation
    dual_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for fundus_batch, oct_batch, labels_batch in paired_val_loader:
            fundus_batch = fundus_batch.to(device)
            oct_batch = oct_batch.to(device)
            labels_batch = labels_batch.long().to(device)
            outputs = dual_model(fundus_batch, oct_batch)
            loss = criterion_dual(outputs, labels_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
    val_loss = val_loss / len(paired_val_loader)
    val_acc = 100 * correct / total
    dual_val_losses.append(val_loss)
    dual_val_accuracies.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], FinalDualModel "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Plot and save Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), dual_train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, num_epochs + 1), dual_val_accuracies, label='Validation Accuracy', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('FinalDualModel Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("FinalDualModel_accuracy.png")
plt.show()
plt.close()

# Plot and save Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), dual_train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs + 1), dual_val_losses, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('FinalDualModel Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("FinalDualModel_loss.png")
plt.show()
plt.close()

# Confusion Matrix and Classification Metrics for Binary Classification
class_names = ['Negative', 'Positive']  # Adjust to match your dataset (e.g., ['Normal', 'Disease'])

# Collect predictions and true labels from the validation set
dual_model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for fundus_batch, oct_batch, labels_batch in paired_val_loader:
        fundus_batch = fundus_batch.to(device)
        oct_batch = oct_batch.to(device)
        labels_batch = labels_batch.long().to(device)
        outputs = dual_model(fundus_batch, oct_batch)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - FinalDualModel (Binary Classification)')
plt.tight_layout()
plt.savefig("FinalDualModel_confusion_matrix_binary.png")
plt.show()
plt.close()

# Compute and display classification metrics
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("\nClassification Report:")
print(report)

# Save classification report to a text file
with open('FinalDualModel_classification_report_binary.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)
