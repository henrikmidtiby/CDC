import cv2
import rasterio
from rasterio.transform import Affine


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

        self.tile_number = None
        self.img = None

    def save_tile(self, output_tile_location):
        print(self.tile_number)
        name_mahal_results = f"{output_tile_location}/mahal{self.tile_number:05d}.tiff"
        if len(self.img.shape) == 2:
            img_to_save = self.img
            channels = 1
            temp_to_save = img_to_save.reshape(1, img_to_save.shape[0], img_to_save.shape[1])
        else:
            img_to_save = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            channels = img_to_save.shape[2]
            temp_to_save = img_to_save.transpose(2, 0, 1)

        new_dataset = rasterio.open(
            name_mahal_results,
            "w",
            driver="GTiff",
            res=self.resolution,
            height=self.size[0],
            width=self.size[1],
            count=channels,
            dtype=temp_to_save.dtype,
            crs=self.crs,
            transform=self.transform,
        )
        new_dataset.write(temp_to_save)
        new_dataset.close()
