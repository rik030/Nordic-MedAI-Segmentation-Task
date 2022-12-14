import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
#import tifffile as tif
#from sklearn.model_selection import train_test_split
import shutil
import random
#size=(512,512)
# def create_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

#         idx = 0
#         for i, m in zip(images, masks):
#             i = cv2.resize(i, size)
#             m = cv2.resize(m, size)

#             tmp_image_name = f"{image_name}_{idx}.jpg"
#             tmp_mask_name  = f"{mask_name}_{idx}.jpg"

#             image_path = os.path.join(save_path, "image/", tmp_image_name)
#             mask_path  = os.path.join(save_path, "mask/", tmp_mask_name)

#             cv2.imwrite(image_path, i)
#             cv2.imwrite(mask_path, m)

#             idx += 1

def load_data(path):
    """ Load all the data and then split them into train and valid dataset. """
    img_path = glob(os.path.join(path, "images/*"))
    msk_path = glob(os.path.join(path, "masks/*"))

    img_path.sort()
    msk_path.sort()

    len_ids = len(img_path)
    train_size = int((80/100)*len_ids)
    valid_size = int((10/100)*len_ids)		## Here 10 is the percent of images used for validation
    test_size = int((10/100)*len_ids)		## Here 10 is the percent of images used for testing
    
    #added by us
    root_dir = '/content/drive/MyDrive/Medai/MedAI_2021_Polyp_Segmentation_Development_Dataset'
    classes_dir = ['/images', '/masks']
    
    #val_ratio = 0.1
    test_ratio = 0.2
    
    for cls in classes_dir:
        # Creating partitions of the data after shuffeling
        src = root_dir + cls # Folder to copy images from
    
        allFileNames = os.listdir(src)
       
        #np.random.shuffle(allFileNames)
        #print("sfsf",len(allFileNames))
        train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - test_ratio))])
        #val_FileNames,
        test_FileNames, val_FileNames = np.split(np.array(test_FileNames),[int(len(test_FileNames)* 0.5)])

        train_FileNames =[src+'/'+ name for name in train_FileNames.tolist()]
        val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
        test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
    
        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))
        print('Testing: ', len(test_FileNames))
    
        # Copy-pasting images
        for name in train_FileNames:
            # image=cv2.imread(name)
            # image=cv2.resize(image,size)
            shutil.copy(name, '/content/drive/MyDrive/Medai/Unet/data' +'/training' + cls)
    
        for name in val_FileNames:
            # image=cv2.imread(name)
            # image=cv2.resize(image,size)
            shutil.copy(name, '/content/drive/MyDrive/Medai/Unet/data' +'/valid' + cls)
    
        for name in test_FileNames:
            # image=cv2.imread(name)
            # image=cv2.resize(image,size)
            shutil.copy(name, '/content/drive/MyDrive/Medai/Unet/data' +'/test' + cls)
    
#     # train_x, test_x = train_test_split(img_path, test_size=test_size, random_state=42)
#     # train_y, test_y = train_test_split(msk_path, test_size=test_size, random_state=42)
#     # train_x, test_x, train_y, test_y = train_test_split(img_path, msk_path, test_size=test_size, random_state=42)

#     """train_x, valid_x = train_test_split(train_x, test_size=test_size, random_state=42)
#     train_y, valid_y = train_test_split(train_y, test_size=test_size, random_state=42)
#     train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=test_size, random_state=42)""" 

     #return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

## cvc segmentation
def get_cvc_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "training", "masks", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "valid", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "valid", "masks", "*.png")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "test", "masks", "*.png")))

    return (train_x, train_y) ,(valid_x, valid_y), (test_x, test_y)  

def main():
    np.random.seed(42)
    path = "/content/drive/MyDrive/Medai/MedAI_2021_Polyp_Segmentation_Development_Dataset"
    new_path="/content/drive/MyDrive/Medai/Unet/data"
    load_data(path)
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = get_cvc_data(new_path)
    #print ("train len",len(train_x))
    #print("test", len(test_x))
    #, (valid_x, valid_y)
    #print(train_x[0])
    #print(test_x[0])
    
    #augment_data(train_x, train_y, "new_data/train/", augment=True)
    #augment_data(valid_x, valid_y, "new_data/valid/", augment=False)
    #augment_data(test_x, test_y, "new_data/test/", augment=False)

if __name__ == "__main__":
    main()