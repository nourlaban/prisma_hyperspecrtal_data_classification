import os.path
import random
import math
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
import tifffile as tiff
from keras.callbacks import ModelCheckpoint
from patches import get_patches
from unet    import unet_model


"""
returns image normalized to be in [-1, 1]
"""
def normalize(img):
    minv = img.min()
    maxv = img.max()
    return 2.0 * (img - minv) / (maxv - minv) - 1.0


NB_BANDS      = 9     # 8 band WorldView imagery
NB_CLASSES    = 10     # buildings, roads, trees, crops, water
CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3,0.1,0.1,0.1,0.1,0.1]
NB_EPOCHS     = 50
BATCH_SIZE    = 64  
UPCONV        = True
PATCH_SIZE    = 128   # should be divisible by 16
NB_TRAIN      = 1200
NB_VAL        = 300
''''''
def read_mask_with_dynamic_classes(mask_path):

    mask = tiff.imread(mask_path)
    unique_classes = np.unique(mask)
    
    height, width = mask.shape
    mask_one_hot = np.zeros((height, width, NB_CLASSES), dtype=np.float32)
    
    for cls in unique_classes:
        if cls != 100:
            mask_one_hot[:, :, cls] = (mask == cls).astype(np.float32)
    
    if 100 in unique_classes:
        mask_one_hot[:, :, 9] = (mask == 100).astype(np.float32)
    
    return mask_one_hot

def get_model():
    return unet_model(NB_CLASSES, PATCH_SIZE, nb_channels=NB_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


data_path = 'data/'
weights_path = 'weights/'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path = os.path.join(weights_path, 'unet_weights.hdf5')
train_ids = [str(i).zfill(2) for i in range(1, 91)]  # all image ids: from 01 to 24


if __name__ == '__main__':

    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VAL   = dict()
    Y_DICT_VAL   = dict()

    print("Reading images")
    for img_id in train_ids:
        img_path = os.path.join(data_path, 'tiles_combined3/{}.tif'.format(img_id))
        mask_path = os.path.join(data_path, 'tiles_shape3/{}.tif'.format(img_id))

        img_m = normalize(tiff.imread(img_path).transpose([1, 0, 2]))
        mask = read_mask_with_dynamic_classes(mask_path)

        print(f"DEBUG: {img_id} - Image shape: {img_m.shape}, Mask shape: {mask.shape}, PATCH_SIZE: {PATCH_SIZE}")

        train_size = int(3/4 * img_m.shape[0])
        X_DICT_TRAIN[img_id] = img_m[:train_size, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_size, :, :]
        X_DICT_VAL[img_id] = img_m[train_size:, :, :]
        Y_DICT_VAL[img_id] = mask[train_size:, :, :]
        
        print(f"Read {img_id}")

    print("Done reading images")
    '''
    for img_id in train_ids:
        img_m = normalize(tiff.imread(os.path.join(data_path, 'tiles_combined3/{}.tif'.format(img_id))).transpose([1, 0, 2]))
        mask = tiff.imread(os.path.join(data_path, 'tiles_shape3/{}.tif'.format(img_id)))
        
        print("DEBUG:", "img_m.shape =", img_m.shape, " mask.shape =", mask.shape)
        
        
        # use 75% of image for training and 25% for validation
        train_size = int(3/4 * img_m.shape[0])
        X_DICT_TRAIN[img_id] = img_m[:train_size, :, :]
        Y_DICT_TRAIN[img_id] = mask [:train_size, :, :]
        X_DICT_VAL[img_id]   = img_m[train_size:, :, :]
        Y_DICT_VAL[img_id]   = mask [train_size:, :, :]
        
        print("Read " + img_id)

    print("Done reading images")

    '''
    def train_net():
        print("Started training")
        
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, NB_TRAIN, PATCH_SIZE)
        x_val,   y_val   = get_patches(X_DICT_VAL,   Y_DICT_VAL,   NB_VAL,   PATCH_SIZE)
        
        model = get_model()
        
        # load saved weights
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('all_labels.csv', append=True, separator=',')
    #     tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCHS,
                verbose=2, shuffle=True, callbacks=[model_checkpoint, csv_logger],
                validation_data=(x_val, y_val))
        
        return model


    train_net()