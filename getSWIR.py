import h5py
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

# Open HDF5 file and read data
filename = r'D:\Narss\Hyperspectral Data\PRS_L2D_STD_20200725083506_20200725083510_0001\PRS_L2D_STD_20200725083506_20200725083510_0001.he5'
file = h5py.File(filename, 'r')

# Path to SWIR data in the HDF5 file (update this if needed)
dataset2_path = '/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'
data2 = file[dataset2_path]

# Define the transform (update the origin and pixel size as needed)
transform = from_origin(
    659247.5,  # X coordinate of the upper-left corner (easting)
    2743007.5,  # Y coordinate of the upper-left corner (northing)
    (698097.4382495880126953 - 659247.5) / data2.shape[2],  # Pixel size in X direction
    (2743007.5 - 2706767.5) / data2.shape[0]  # Pixel size in Y direction
)

# CRS definition (update this with actual CRS information if different)
crs_string = CRS.from_epsg(32636)  # Assuming EPSG:32636 (UTM Zone 36N), update if necessary

# Metadata for rasterio
metadata = {
    'driver': 'GTiff',
    'height': data2.shape[0],
    'width': data2.shape[2],
    'count': data2.shape[1],  # Number of bands
    'dtype': str(data2.dtype),
    'crs': crs_string,
    'transform': transform
}

# Output GeoTIFF file path
tif_path = r'D:\Narss\Hyperspectral Data\final output\stacked_hyperspectral_image_SWIR.tif'

# Write data to GeoTIFF
with rasterio.open(tif_path, 'w', **metadata) as dst:
    for i in range(data2.shape[1]):
        dst.write(data2[:, i, :], i + 1)  # Write each band

# Close the HDF5 file
file.close()
