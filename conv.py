import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.crs import CRS

# Path to the shapefile
shapefile_path = r'D:\Narss\Hyperspectral Data\Abu Rhusheid_Train\Abu_Rhusheid_train.shp'
vector_data = gpd.read_file(shapefile_path)

# Define raster parameters
pixel_size = 30  # Define the pixel size (resolution) in map units
output_raster_path = r'D:\Narss\Hyperspectral Data\final output\raster_categorical2.tif'

# Check and transform CRS if needed
if vector_data.crs is None:
    raise ValueError("The shapefile does not have a CRS defined.")

# Optionally, reproject to a common CRS (if needed)
# vector_data = vector_data.to_crs(CRS.from_epsg(32636))
dataset1_path=r'D:\Narss\Hyperspectral Data\final output\stacked_hyperspectral_image_VNIR9.tif'

with rasterio.open(dataset1_path, 'r') as dst:
 imdata=dst.read(6)
 width = int(dst.width)
 height = int(dst.height)

nodataval=0
nodata_mask=imdata==nodataval

# Get the bounds of the vector data
bounds = vector_data.total_bounds


# Ensure width and height are greater than zero
if width <= 0 or height <= 0:
    raise ValueError(f"Invalid raster dimensions: width={width}, height={height}")

# Define the transform
transform = from_origin(659247.5, 2743007.5, (698097.4382495880126953 - 659247.5) /  width, (2743007.5 - 2706767.5) /  height)

# Define CRS (update if necessary to match vector_data's CRS)
crs = vector_data.crs.to_string()  # Use the CRS from the shapefile

# Choose the categorical attribute to rasterize
attribute = 'Classvalue'  # Replace with your actual attribute name

# Check if the attribute exists
if attribute not in vector_data.columns:
    raise ValueError(f"Attribute '{attribute}' not found in shapefile.")

# Extract unique categories
categories = vector_data[attribute].unique()
category_map = {category: idx + 1 for idx, category in enumerate(categories)}

print(f"Categories and corresponding values: {category_map}")

# Rasterize the vector data
metadata = {
    'driver': 'GTiff',
    'count': 1,  # Single band raster
    'dtype': 'uint8',
    'width': width,
    'height': height,
    'crs': crs,
    'transform': transform
}

try:
    with rasterio.open(output_raster_path, 'w', **metadata) as dst:
        out_image = rasterize(
            [(geometry, category_map[value]) 
             for value, geometry in zip(vector_data[attribute], vector_data.geometry)],
            out_shape=(height, width),
            transform=transform,
            fill=0,  # Background value
            dtype='uint8'
        )
        out_image[nodata_mask]=100
        dst.write(out_image, 1)  # Write the rasterized data to the first band
       
    print(f"Shapefile rasterized with categorical data and saved as {output_raster_path}")

except rasterio.errors.RasterioIOError as e:
    print(f"RasterioIOError: {e}")

except Exception as e:
    print(f"An error occurred: {e}")
