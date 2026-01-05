import os
import time
import datetime
import random
import numpy as np
from glob import glob
import albumentations as A
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from model.UNet import UNet

from metrics import Metrics,eval
from utils import (
    seeding, shuffling, create_dir, init_mask,
    epoch_time, rle_encode, rle_decode, print_and_save, load_data,load_data2
    )
from loss import DiceBCELoss




class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size
    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = image.astype(np.float32)


        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = mask.astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples


def train(model, loader,  optimizer, loss_fn, device):
    epoch_loss = 0
    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)

    return epoch_loss

def evaluate(model, loader, loss_fn, device,total_batch):
    epoch_loss = 0
    model.eval()
    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean','Dice'])
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = torch.sigmoid(model(x))
            _recall, _specificity, _precision, _F1, _F2, \
                _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean,_Dice = eval(y_pred, y, 0.5)

            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1=_F1, F2=_F2, ACC_overall=_ACC_overall, IoU_poly=_IoU_poly,
                           IoU_bg=_IoU_bg, IoU_mean=_IoU_mean,Dice=_Dice
                           )
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

    metrics_result = metrics.mean(total_batch)
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss,metrics_result

if __name__ == "__main__":

    ##########################################################
    #Code explanation
    #This code is used to obtain the 7 training result files for each layer.
    #To obtain the training results of a specific layer, it is necessary to set the learning rate of that layer to lr, while setting the learning rate of the other layers to 1e-4.
    ##########################################################

    for virdataset in ["ClinicDB"]:
        for lr in [0.5*1e-4,0.5**2*1e-4,0.5**3*1e-4, 1e-4, 1.5*1e-4,1.5**2*1e-4,1.5**3*1e-4]:
                k_fold = 1
                index = 2
                seeding(index) #2 4 6 8
                model = UNet(classes=1)
                opt = [torch.optim.Adam([
                    
                    {'params': model.inc.parameters(), 'lr': lr},#Layer 1
                    {'params': model.down1.parameters(), 'lr': lr},#Layer 1

                    #############################################
                    {'params': model.down2.parameters(), 'lr': 1e-4},#Layer 2

                    {'params': model.down3.parameters(), 'lr': 1e-4}, #Layer 3

                    {'params': model.down4.parameters(), 'lr': 1e-4}, #Layer 4

                    {'params': model.up1.parameters(), 'lr': 1e-4},  #Layer 5

                    {'params': model.up2.parameters(), 'lr': 1e-4},  #Layer 6

                    {'params': model.up3.parameters(), 'lr': 1e-4},#Layer 7

                    {'params': model.up4.parameters(), 'lr': 1e-4},  #Layer 8

                    {'params': model.outc.parameters(), 'lr': 1e-4}, #Layer 8


                ])]

                size = (256, 256)
                batch_size = 8
                num_epochs = 6
                for indexOPT, optimizer in enumerate(opt):
                    dataset = virdataset
                    modelName = "UNet_" + str(indexOPT) +"_"+str(lr) # DCSAU MSRAformer UNetPlusPlus  ACC_UNet  caranet EUNet
                    results_file = UNet + "_"+lr  # 2018res  kva  Cli
                    checkpoint_path = results_file + "/"+modelName+"checkpoint.pth"
                    ################################################


                    """ Directories """
                    create_dir(results_file)

                    """ Training logfile """
                    train_log_path = results_file+"/"+modelName+"train_log.txt"
                    if os.path.exists(train_log_path):
                        print("Log file exists")
                    else:
                        train_log = open(results_file+"/"+modelName+"train_log.txt", "w")
                        train_log.write("\n")
                        train_log.close()

                    """ Record Date & Time """
                    datetime_object = str(datetime.datetime.now())
                    print_and_save(train_log_path, datetime_object)
                    data_str = f"optimizer: {indexOPT}, Learning Rate: {lr}\n"
                    print_and_save(train_log_path, data_str)

                    """ Dataset """
                    path = dataset
                    (train_x, train_y), (valid_x, valid_y) = load_data2(path,k_fold)
                    train_x, train_y = shuffling(train_x, train_y)
                    val_total_batch = int(len(valid_x) / 1)
                    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
                    print_and_save(train_log_path, data_str)

                    """ Data augmentation: Transforms """
                    transform =  A.Compose([
                        A.Rotate(limit=35, p=0.3),
                        A.HorizontalFlip(p=0.3),
                        A.VerticalFlip(p=0.3),
                        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
                    ],is_check_shapes=False)#########跑新的需要加

                    """ Dataset and loader """
                    train_dataset = DATASET(train_x, train_y, size, transform=transform)
                    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

                    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
                    print_and_save(train_log_path, data_str)

                    train_loader = DataLoader(
                        dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2
                    )

                    valid_loader = DataLoader(
                        dataset=valid_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=2
                    )

                    """ Model """
                    device = torch.device('cuda')
                    model = model.to(device)


                    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
                    loss_fn = DiceBCELoss()
                    loss_name = "BCE Dice Loss"

                    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
                    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
                    print_and_save(train_log_path, data_str)

                    """ Training the model1. """
                    best_valid_Iou = 0.0

                    for epoch in range(num_epochs):
                        start_time = time.time()

                        train_loss = train(model, train_loader,  optimizer, loss_fn, device)
                        valid_loss, metrics_result = evaluate(model, valid_loader,  loss_fn, device, val_total_batch)
                        # scheduler.step(valid_loss)

                        if metrics_result['IoU_mean'] > best_valid_Iou:
                            print("save model, current Iou is "+str(metrics_result['IoU_mean']))
                            best_valid_Iou = metrics_result['IoU_mean']
                            data_str = f"Saving checkpoint: {checkpoint_path}"
                            print_and_save(train_log_path, data_str)
                            torch.save(model.state_dict(), checkpoint_path)

                        end_time = time.time()
                        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
                        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
                        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
                        print_and_save(train_log_path, data_str)
                        print_and_save(train_log_path, f'\trecall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f,'
                              ' F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, Dice: %.4f \n'
                              % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                                 metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
                                 metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean'],metrics_result['Dice']))
                    datetime_object = str(datetime.datetime.now())
                    print_and_save(train_log_path, datetime_object)