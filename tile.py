import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window


class Tile:
    def __init__(self, start_point, position, height, width, resolution, crs, left, top):
        # Data for the tile
        self.size = (height, width)
        self.tile_position = position
        self.ulc = start_point
        self.lrc = (start_point[0] + height, start_point[1] + width)
        self.processing_range = [[0, 0], [0, 0]]
        self.resolution = resolution
        self.crs = crs
        self.left = left
        self.top = top
        self.ulc_global = [
            self.top - (self.ulc[0] * self.resolution[0]),
            self.left + (self.ulc[1] * self.resolution[1]),
        ]
        self.transform = Affine.translation(
            self.ulc_global[1] + self.resolution[0] / 2, self.ulc_global[0] - self.resolution[0] / 2
        ) * Affine.scale(self.resolution[0], -self.resolution[0])

    def read_tile(self, orthomosaic_filename):
        with rasterio.open(orthomosaic_filename) as src:
            window = Window.from_slices(
                (self.ulc[0], self.lrc[0]),
                (self.ulc[1], self.lrc[1]),
            )
            img = src.read(window=window)
            mask = src.read_masks(window=window)
        return img, mask

    def save_tile(self, image, output_tile_location):
        output_tile_filename = f"{output_tile_location}/{self.tile_number:05d}.tiff"
        new_dataset = rasterio.open(
            output_tile_filename,
            "w",
            driver="GTiff",
            res=self.resolution,
            height=self.size[0],
            width=self.size[1],
            count=image.shape[0],
            dtype=image.dtype,
            crs=self.crs,
            transform=self.transform,
        )
        new_dataset.write(image)
        new_dataset.close()
