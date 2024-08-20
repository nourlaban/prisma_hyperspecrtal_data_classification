import os
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

# Paths to the input GeoTIFF files
rasterized_tiff_path = r'D:\Narss\Hyperspectral Data\Abu Rhusheid_Train\raster_categorical2.tif'
combined_tiff_path = r'D:\Narss\Hyperspectral Data\combined_hyperspectral_image_VNIR_SWIR.tif'

# Output directories for tiles
output_dirs = {
    'rasterized': r'D:\Narss\tiles_shape1',
    'combined': r'D:\Narss\tiles_combined'
}

# Ensure the output directories exist
for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

# Tile size
tile_height = 64
tile_width = 64

def create_tiles(tiff_path, output_dir):
    # Open the GeoTIFF file
    with rasterio.open(tiff_path) as src:
        # Get metadata and transform from the source file
        metadata = src.meta.copy()
        
        # Get the number of bands in the dataset
        num_bands = src.count

        # Calculate the transform using from_origin (modify values as needed)
        transform = from_origin(
            659247.5, 
            2743007.5, 
            (698097.4382495880126953 - 659247.5) / src.width, 
            (2743007.5 - 2706767.5) / src.height
        )
        metadata.update({"transform": transform})

        # Loop through the data and create tiles
        for row in range(0, src.height, tile_height):
            for col in range(0, src.width, tile_width):
                # Define the window for the tile
                window = Window(col, row, tile_width, tile_height)

                # Read all bands for this window
                tile = src.read(window=window)

                # Ensure the tile has the correct dimensions and all bands
                if tile.shape[1] == tile_height and tile.shape[2] == tile_width and tile.shape[0] == num_bands:
                    # Create metadata for this tile
                    tile_metadata = metadata.copy()
                    tile_metadata.update({
                        'height': tile_height,
                        'width': tile_width,
                        'count': num_bands,  # Ensure the number of bands is correct
                        'transform': rasterio.windows.transform(window, src.transform)
                    })

                    # Define the output filename for the tile
                    tile_filename = f'tile_{row}_{col}.tif'

                    # Write the tile to a new GeoTIFF file
                    with rasterio.open(os.path.join(output_dir, tile_filename), 'w', **tile_metadata) as dst:
                        dst.write(tile)

    print(f"Tiled raster saved in {output_dir}")


# Create tiles for both rasterized and combined GeoTIFF files
create_tiles(rasterized_tiff_path, output_dirs['rasterized'])
create_tiles(combined_tiff_path, output_dirs['combined'])
