import os
import cv2
import sys
import numpy as np
sys_path = "/home/cs23b1055/"


sys.path.append(f'{sys_path}functions.py')
from functions import mini, fibonacci, algo3, apply_algo

path ="/home/cs23b1055/oct"

for i in range(1, 3001):
    
    img_path = os.path.join(path, f"DR/{i}.jpg")
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224,224))
    
    features = apply_algo(image)
    features = np.transpose(features, (2, 0, 1))
    
    os.remove(img_path)
    
    feat1 = features[0]
    feat2 = features[1]
    feat3 = features[2]
    feat4 = features[3]
    
    cv2.imwrite(f"{path}/DR/{i}_a.jpg", feat1)
    cv2.imwrite(f"{path}/DR/{i}_b.jpg", feat2)
    cv2.imwrite(f"{path}/DR/{i}_c.jpg", feat3)
    cv2.imwrite(f"{path}/DR/{i}_d.jpg", feat4)
    
    print(i, " DR")
    
    
for i in range(1, 3001):
    
    img_path = os.path.join(path, f"NORMAL/{i}.jpg")
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224,224))
    
    features = apply_algo(image)
    features = np.transpose(features, (2, 0, 1))
    
    os.remove(img_path)
    
    feat1 = features[0]
    feat2 = features[1]
    feat3 = features[2]
    feat4 = features[3]
    
    cv2.imwrite(f"{path}/NORMAL/{i}_a.jpg", feat1)
    cv2.imwrite(f"{path}/NORMAL/{i}_b.jpg", feat2)
    cv2.imwrite(f"{path}/NORMAL/{i}_c.jpg", feat3)
    cv2.imwrite(f"{path}/NORMAL/{i}_d.jpg", feat4)
    
    print(i, " NORMAL")
    

    
    
    
    
    
    
    