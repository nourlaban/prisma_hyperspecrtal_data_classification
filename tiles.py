import os
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

# Paths to the input GeoTIFF files
rasterized_tiff_path = r'D:\Narss\Hyperspectral Data\Abu Rhusheid_Train\raster_categorical2.tif'
combined_tiff_path = r'D:\Narss\Hyperspectral Data\final output\pca_hyperspectral_image_uint16_7.tif'

# Output directories for tiles
output_dirs = {
    'rasterized': r'D:\Narss\Hyperspectral Data\final output\tiles_shape1',
    'combined': r'D:\Narss\Hyperspectral Data\final output\tiles_combined'
}

# Label output file
label_output_file = r'D:\Narss\Hyperspectral Data\final output\labels\all_labels.txt'

# Ensure the output directories exist
for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(label_output_file), exist_ok=True)

# Tile size
tile_height = 128
tile_width = 128

def create_tiles_and_labels(raster_tiff_path, pca_tiff_path, output_dirs, label_file):
    # Open the raster categorical and PCA GeoTIFF files
    with rasterio.open(raster_tiff_path) as raster_src, rasterio.open(pca_tiff_path) as pca_src:
        # Get metadata and transform from the raster file
        raster_metadata = raster_src.meta.copy()
        pca_metadata = pca_src.meta.copy()

        # Get the number of bands in the datasets
        num_raster_bands = raster_src.count
        num_pca_bands = pca_src.count

        # Calculate the transform using from_origin (modify values as needed)
        transform = from_origin(
            659247.5, 
            2743007.5, 
            (698097.4382495880126953 - 659247.5) / raster_src.width, 
            (2743007.5 - 2706767.5) / raster_src.height
        )
        raster_metadata.update({"transform": transform})
        pca_metadata.update({"transform": transform})

        # Open the label file for writing
        with open(label_file, 'w') as lbl_file:
            # Loop through the data and create tiles
            for row in range(0, raster_src.height, tile_height):
                for col in range(0, raster_src.width, tile_width):
                    # Define the window for the tile
                    window = Window(col, row, tile_width, tile_height)

                    # Read all bands for this window from both raster and PCA
                    raster_tile = raster_src.read(window=window)
                    pca_tile = pca_src.read(window=window)

                    # Ensure both tiles have the correct dimensions and all bands
                    if (raster_tile.shape[1] == tile_height and raster_tile.shape[2] == tile_width and raster_tile.shape[0] == num_raster_bands and
                        pca_tile.shape[1] == tile_height and pca_tile.shape[2] == tile_width and pca_tile.shape[0] == num_pca_bands):
                        
                        # Create metadata for both tiles
                        raster_tile_metadata = raster_metadata.copy()
                        raster_tile_metadata.update({
                            'height': tile_height,
                            'width': tile_width,
                            'count': num_raster_bands,
                            'transform': rasterio.windows.transform(window, raster_src.transform)
                        })

                        pca_tile_metadata = pca_metadata.copy()
                        pca_tile_metadata.update({
                            'height': tile_height,
                            'width': tile_width,
                            'count': num_pca_bands,
                            'transform': rasterio.windows.transform(window, pca_src.transform)
                        })

                        # Define the output filenames for the tiles
                        raster_tile_filename = f'tile_{row}_{col}_raster.tif'
                        pca_tile_filename = f'tile_{row}_{col}_pca.tif'

                        # Write the tiles to new GeoTIFF files
                        with rasterio.open(os.path.join(output_dirs['rasterized'], raster_tile_filename), 'w', **raster_tile_metadata) as raster_dst:
                            raster_dst.write(raster_tile)

                        with rasterio.open(os.path.join(output_dirs['combined'], pca_tile_filename), 'w', **pca_tile_metadata) as pca_dst:
                            pca_dst.write(pca_tile)

                        # Write correspondence information to the label file
                        lbl_file.write(f"{raster_tile_filename}\t{pca_tile_filename}\n")

    print(f"Tiled rasters saved in {output_dirs['rasterized']} and {output_dirs['combined']}, and correspondence labels saved in {label_file}")

# Create tiles and a single label file for both rasterized and combined GeoTIFF files
create_tiles_and_labels(rasterized_tiff_path, combined_tiff_path, output_dirs, label_output_file)
