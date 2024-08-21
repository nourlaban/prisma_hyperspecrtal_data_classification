import h5py
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin

# Open HDF5 file and read data
filename = r'D:\Narss\Hyperspectral Data\PRS_L2D_STD_20200725083506_20200725083510_0001\PRS_L2D_STD_20200725083506_20200725083510_0001.he5'
file = h5py.File(filename, 'r')
dataset1_path = '/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'
data1 = file[dataset1_path]
# Metadata for rasterio
transform = Affine.translation(0, 0)  # Placeholder, adjust this
crs_string = {'init': 'epsg:32636'}  # Placeholder, update with actual CRS


 # Define the transform (example transform, adjust according to your data)
transform = from_origin(659247.5, 2743007.5, (698097.4382495880126953 - 659247.5) /  data1.shape[2], (2743007.5 - 2706767.5) /  data1.shape[0]) # Adjust the origin and pixel size as needed
metadata = {
    'driver': 'GTiff',
    'height': data1.shape[0],
    'width': data1.shape[2],
    'count': data1.shape[1],
    'dtype': str(data1.dtype),
    'crs': crs_string,
    'transform': transform
}

# Write data to GeoTIFF
tif_path = r'D:\Narss\Hyperspectral Data\final output\stacked_hyperspectral_image_VNIR9.tif'
with rasterio.open(tif_path, 'w', **metadata) as dst:
    for i in range(data1.shape[1]):
        dst.write(data1[:, i, :], i+1)  # Write each band   
# Close the HDF5 file
file.close()