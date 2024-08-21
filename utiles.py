import os
import pickle
import json
from torch.utils.data import DataLoader
import pathlib
from customdatasets import CustomDataSet
from transformations import 
from transformations import MoveAxis, Normalize01, RandomCrop
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from os import walk
import torch as t
import numpy as np
import torch.nn as nn


def get_files(path):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        for names in filenames:
            files.append(dirpath + '/' + names)
    return files

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_model(device, cl):
    
    unet = smp.Unet('resnet152', classes=cl, activation=None, encoder_weights='imagenet')

    if t.cuda.is_available():
        unet.cuda()         
    
    unet = unet.to(device)
    return unet

def import_data(args, batch_sz, set = 'project_3', crop_size):

    root = pathlib.Path('./')
    if set == 'project_1':
        inputs = get_files('./input_data/project_1/image/')
        targets = get_files('./input_data/project_1/target/')

    if set == 'project_2':
        inputs = get_files('./input_data/project_2/image/')
        targets = get_files('./input_data/project_2/target/')

    if set == 'project_3':
        inputs = get_files('./input_data/project_3/image/')
        targets = get_files('./input_data/project_3/target/')

    split = 0.8  

    inputs_train, inputs_valid = train_test_split(
        inputs,
        random_state=42,
        train_size=split,
        shuffle=True)

    targets_train, targets_valid = train_test_split(
        targets,
        random_state=42,
        train_size=split,
        shuffle=True)

    # Add your desired transformations here
    transforms = Compose([
        MoveAxis(),
        Normalize01(),
        RandomCrop(crop_size),
        RandomFlip()
        ])

    # train dataset
    dataset_train = CustomDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        transform=transforms)


    # validation dataset
    dataset_valid = CustomDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        transform=transforms)


    # train dataloader
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=batch_sz,
                                    shuffle=True
                                    )

    # validation dataloader
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                    batch_size=batch_sz,
                                    shuffle=True)
       
    return dataloader_training, dataloader_validation



def checkpoint(f, tag, args, device, dataloader_training, dataloader_validation):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "train": dataloader_training,
        "valid": dataloader_validation

    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)