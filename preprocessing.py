import numpy as np
import cv2



def get_oct_image(key):
    
    dataset_path = "/home/cs23b1055/images/"
    
    oct_feata = f"{dataset_path}/oct/{key}_a.jpg"
    oct_featb = f"{dataset_path}/oct/{key}_b.jpg"
    oct_featc = f"{dataset_path}/oct/{key}_c.jpg"
    oct_featd = f"{dataset_path}/oct/{key}_d.jpg"
    
    oct_feat1 = cv2.imread(oct_feata, cv2.IMREAD_GRAYSCALE)
    oct_feat2 = cv2.imread(oct_featb, cv2.IMREAD_GRAYSCALE)
    oct_feat3 = cv2.imread(oct_featc, cv2.IMREAD_GRAYSCALE)
    oct_feat4 = cv2.imread(oct_featd, cv2.IMREAD_GRAYSCALE)
    
    oct_feat1 = oct_feat1 / 255.0
    oct_feat2 = oct_feat2 / 255.0
    oct_feat3 = oct_feat3 / 255.0
    oct_feat4 = oct_feat4 / 255.0
    
    oct_feat = np.stack([oct_feat1, oct_feat2, oct_feat3, oct_feat4], axis=-1)
    
    return oct_feat
    


def apply_fun_pre(path):

    img_bgr = cv2.imread(path)
    img_bgr = cv2.resize(img_bgr, (224,224))


    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


    v_channel = img_hsv[:, :, 2]


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_clahe = clahe.apply(v_channel)

    img_hsv_clahe = img_hsv.copy()
    img_hsv_clahe[:, :, 2] = v_clahe
    img_clahe_rgb = cv2.cvtColor(img_hsv_clahe, cv2.COLOR_HSV2RGB)
    
    return img_clahe_rgb


def fibonacci(n):
    
    
    output = []
    
    numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    numbers.reverse()
    
    
    for num in numbers:
        if num > n:
            output.append(0)
        else:
            n = n - num
            output.append(num)
    
    output.reverse()
    
    return output

def mini(features, n):
    
    
    mini = 255 * 500 * 750
    index = 0
    
    for i in range(0, n):
        if np.sum(features[:, :, i]) < mini:
            mini = np.sum(features[:, :, i])
            index = i
    
    return index            
    
    

def algo3(features):
    
    n = 12
    
    for k in range(0, 6):
        index = mini(features, n)
        features = np.delete(features, index, axis=2)
        n -= 1
        
    return features[:,:,2:]    


def apply_clahe(image):
    
    # Check the depth of the input image
    if image.dtype == np.float64:
        # Normalize and convert to 8-bit unsigned integer (CV_8U)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    elif image.dtype != np.uint8:
        # Convert other types (e.g., float32) to uint8 if needed
        image = image.astype(np.uint8)

    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create a CLAHE object (clipLimit and tileGridSize can be tuned)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE
    clahe_image = clahe.apply(gray)
    
    # If the original image was color, convert back to 3 channels
    if len(image.shape) == 3:
        clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    
    return clahe_image

def apply_algo(image):
    
    
    features = []

    for i in range(0, image.shape[0]):
        row = []
        for j in range(0, image.shape[1]):
            row.append(fibonacci(image[i][j]))
        features.append(row)
        
        
    features = np.array(features)    
    features = algo3(features)
    
    return features