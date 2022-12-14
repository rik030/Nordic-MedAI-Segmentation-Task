import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, MotionBlur, MedianBlur, RandomBrightnessContrast, RandomGamma, ChannelShuffle

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):

    train_x = sorted(glob(os.path.join(path, "training", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "training", "masks", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "valid", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "valid", "masks", "*.png")))

    return (train_x, train_y), (valid_x, valid_y)

def preprocess_data(images, masks, save_path, augment=True):

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting names """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x)
        y = cv2.imread(y)
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4, y5]

        else:
            X = [x]
            Y = [y]
        index = 0
        for i, m in zip(X, Y):
            if len(X) == 1:
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{index}.jpg"
                tmp_mask_name = f"{name}_{index}.jpg"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
              #raise Exception("Could not write image")
            index += 1


if __name__ == "__main__":
    np.random.seed(42)

    data_path = "/content/drive/MyDrive/MedAi/UNet/data"
    (train_x, train_y), (valid_x, valid_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(valid_x)} - {len(valid_y)}")

    preprocess_data(train_x, train_y, "/content/drive/MyDrive/MedAi/UNet_LD/dataset/train", augment=True)
    preprocess_data(valid_x, valid_y, "/content/drive/MyDrive/MedAi/UNet_LD/dataset/valid", augment=False)

    
