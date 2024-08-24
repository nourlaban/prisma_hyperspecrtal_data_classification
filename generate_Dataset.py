import h5py
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

class HyperspectralImageReader:
    def __init__(self, filename):
        self.filename = filename
        self.h5file = None
        self.datasetVNIR = None
        self.datasetSWIR = None
        self.metadata = None

    def open_file(self):
        self.h5file = h5py.File(self.filename, 'r')
        datasetVNIR_path = '/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'
        datasetSWIR_path = '/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'
        self.datasetVNIR = self.h5file[datasetVNIR_path]
        self.datasetSWIR = self.h5file[datasetSWIR_path]

    def set_metadata(self, origin_x, origin_y, pixel_size_x, pixel_size_y):
        transform = from_origin(origin_x, origin_y, pixel_size_x, pixel_size_y)
        vnir_bands = self.datasetVNIR.shape[1] - 5
        swir_bands = self.datasetSWIR.shape[1] - len(self.excluded_swir_bands())
        total_bands = vnir_bands + swir_bands
        self.metadata = {
            'driver': 'GTiff',
            'height': self.datasetVNIR.shape[0],
            'width': self.datasetVNIR.shape[2],
            'count': total_bands,
            'dtype': str(self.datasetVNIR.dtype),
            'crs': CRS.from_epsg(32636),  # Use CRS object instead of dict
            'transform': transform
        }

    def excluded_swir_bands(self):
        return list(range(3, 6)) + list(range(40, 57)) + list(range(86, 112)) + list(range(152, 172))

    def write_to_geotiff(self, output_path):
        with rasterio.open(output_path, 'w', **self.metadata) as dst:
            # Write VNIR bands
            for i in range(self.datasetVNIR.shape[1] - 5):
                dst.write(self.datasetVNIR[:, i + 5, :], i + 1)
            
            # Write SWIR bands, excluding specified ones
            current_band = self.datasetVNIR.shape[1] - 5
            for j in range(self.datasetSWIR.shape[1]):
                if j not in self.excluded_swir_bands():
                    dst.write(self.datasetSWIR[:, j, :], current_band + 1)
                    current_band += 1

    def close(self):
        self.h5file.close()

# Example usage
h5_file = r"C:\Users\utente\Desktop\narss\Hyperspectral Data\PRS_L2D_STD_20200725083506_20200725083510_0001\PRS_L2D_STD_20200725083506_20200725083510_0001.he5"
tif_file = r"C:\Users\utente\Desktop\narss\Hyperspectral Data\PRS_L2D_STD_20200725083506_20200725083510_0001\stacked_hyperspectral_image_VNIR.tif"

reader = HyperspectralImageReader(h5_file)
reader.open_file()
reader.set_metadata(
    origin_x=659247.5,
    origin_y=2743007.5,
    pixel_size_x=(698097.4382495880126953 - 659247.5) / reader.datasetVNIR.shape[2],
    pixel_size_y=(2743007.5 - 2706767.5) / reader.datasetVNIR.shape[0]
)
reader.write_to_geotiff(r"C:\Users\utente\Desktop\narss 2\generateddataset.tif")
reader.close()
