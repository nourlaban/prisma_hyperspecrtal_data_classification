import os
import numpy as np
import tifffile as tiff
from train import get_model, NB_CLASSES
from patches import DEFAULT_PATCH_SIZE
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
import math

# Function to normalize image data
def normalize(img):
    minv = img.min()
    maxv = img.max()
    return 2.0 * (img - minv) / (maxv - minv) - 1.0

# Prediction function to process the image in patches
def predict(x, model, patch_size=128, nb_classes=NB_CLASSES):
    img_height = x.shape[0]
    img_width = x.shape[1]
    nb_channels = x.shape[2]
    
    # Determine the number of patches to process
    nb_patches_vertical = math.ceil(img_height / patch_size)
    nb_patches_horizontal = math.ceil(img_width / patch_size)
    
    # Extend the image if necessary to fit into patch sizes
    extended_height = patch_size * nb_patches_vertical
    extended_width = patch_size * nb_patches_horizontal
    ext_x = np.zeros((extended_height, extended_width, nb_channels), dtype=np.float32)
    
    # Fill the extended image with the original image and mirrored borders
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]
    
    # Divide the image into patches
    patches_list = []
    for i in range(nb_patches_vertical):
        for j in range(nb_patches_horizontal):
            x0, x1 = i * patch_size, (i + 1) * patch_size
            y0, y1 = j * patch_size, (j + 1) * patch_size
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    
    # Convert patches to a numpy array and predict
    patches = np.asarray(patches_list)
    patches_predict = model.predict(patches, batch_size=4)
    
    # Combine the patches back into the full image prediction
    prediction = np.zeros((extended_height, extended_width, nb_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // nb_patches_horizontal
        j = k % nb_patches_horizontal
        x0, x1 = i * patch_size, (i + 1) * patch_size
        y0, y1 = j * patch_size, (j + 1) * patch_size
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    
    return prediction[:img_height, :img_width, :]  # Return the original image size
# Picture from mask function
# Picture from mask function
def picture_from_mask(mask, threshold=0):
    # Colors corresponding to classes (assuming 0-8)
    colors = {
        0: [150, 150, 150],   # buildings
        1: [223, 194, 125],   # roads & tracks
        2: [27,  120, 55],    # trees
        3: [166, 219, 160],   # crops
        4: [116, 173, 209],   # water
        5: [255, 0, 0],       # red
        6: [255, 255, 0],     # yellow
        7: [0, 0, 139],       # dark blue
        8: [150, 150, 150]    # buildings
    }
    
    # Initialize the output image
    pict = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)

    # Assign colors based on the mask class
    for i in range(9):  # Loop through all classes
        class_mask = mask[i, :, :] > threshold
        for color_index in range(3):  # RGB channels
            pict[:, :, color_index][class_mask] = colors[i][color_index]

    return pict

# Main function to load images, predict, and save results
def main():
    model = get_model()  # Assuming get_model() loads the trained model
    data_dir = 'data/tiles_combined3'  # Input directory containing the images
    results_dir = 'results/'  # Output directory to save predictions

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Iterate through each image in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.tif'):
            image_path = os.path.join(data_dir, filename)
            
            # Load and normalize the image
            img = normalize(tiff.imread(image_path).transpose([1, 0, 2]))
            print(f"Processing {filename}, image shape: {img.shape}")
            
            # Predict using the model
            res = predict(img, model, patch_size=DEFAULT_PATCH_SIZE, nb_classes=NB_CLASSES).transpose([2, 0, 1])
            print(f"Prediction result shape: {res.shape}")

            # Exclude the last band (9th band)
            res = res[:-1]  # Keep only the first 8 bands
            
            # Generate the color image
            res_map = picture_from_mask(res, threshold=0.5)

            # Save the resulting RGB image as a TIFF file
            output_path = os.path.join(results_dir, f'result_{filename[:-4]}.tif')
            tiff.imwrite(output_path, res_map, photometric='rgb')  # Specify photometric interpretation
            print(f"Saved the result as {output_path}")

if __name__ == "__main__":
    main()
