import rasterio
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the hyperspectral image using rasterio
def load_hyperspectral_image(image_path):
    with rasterio.open(image_path) as dataset:
        hyperspectral_data = dataset.read().transpose((1, 2, 0))
        profile = dataset.profile
    return hyperspectral_data, profile

# Apply PCA to the hyperspectral image
def apply_pca(hyperspectral_data, n_components=7):
    h, w, bands = hyperspectral_data.shape
    reshaped_data = hyperspectral_data.reshape(-1, bands)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped_data)
    
    pca_image = pca_result.reshape(h, w, n_components)
    return pca_image

# Normalize and convert to UInt16
def convert_to_uint16(pca_image):
    # Normalize the data to range [0, 1]
    min_val = pca_image.min()
    max_val = pca_image.max()
    normalized_image = (pca_image - min_val) / (max_val - min_val)
    
    # Scale to range [0, 65535] and convert to UInt16
    uint16_image = (normalized_image * 65535).astype(np.uint16)
    return uint16_image

# Visualize the PCA result
def visualize_pca(pca_image):
    plt.figure(figsize=(10, 10))
    plt.imshow(pca_image[:, :, :3])
    plt.title('PCA Applied Hyperspectral Image')
    plt.show()

# Save the PCA-transformed image in UInt16 format
def save_pca_image(uint16_image, output_path, original_profile):
    new_profile = original_profile.copy()
    new_profile.update({
        'count': uint16_image.shape[2],
        'dtype': 'uint16'
    })
    
    with rasterio.open(output_path, 'w', **new_profile) as dst:
        for i in range(uint16_image.shape[2]):
            dst.write(uint16_image[:, :, i], i + 1)

# Main function
if __name__ == '__main__':
    image_path = r'd:\Narss\Hyperspectral Data\final output\combined_hyperspectral_image_VNIR_SWIR2.tif'
    output_path = r'd:\Narss\Hyperspectral Data\final output\pca_hyperspectral_image_uint16_7.tif'
    
    # Load the hyperspectral data
    hyperspectral_data, profile = load_hyperspectral_image(image_path)
    
    # Apply PCA to reduce dimensionality
    pca_image = apply_pca(hyperspectral_data, n_components=7)
    
    # Normalize and convert to UInt16
    uint16_image = convert_to_uint16(pca_image)
    
    # Visualize the PCA result (note: this will still display as float32)
    #visualize_pca(uint16_image.astype(np.float32) / 65535)
    
    # Save the PCA-transformed image in UInt16 format
    save_pca_image(uint16_image, output_path, profile)
