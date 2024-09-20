import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import CheckButtons, Slider, Button
from tkinter import Tk, filedialog
import threading
import argparse

class RasterAnnotator:
    def __init__(self, raster_file,alphachannel=True):
        self.raster_file = raster_file
        self.dataset = rasterio.open(raster_file)
        match alphachannel:
            case True: 
                self.bands = self.dataset.count-1
            case False:
                self.bands=self.dataset.count
            case _:
                self.bands=self.dataset.count
        
        self.selected_bands = np.ones(self.bands, dtype=bool)  # Initially all bands are selected
        self.fig, self.ax = plt.subplots()
        self.annotated_mask = None
        self.brush_size = 0 # Initial brush size
        self.update_lock = threading.Lock()

        # Create check buttons for selecting bands
        self.check_ax = plt.axes([0.05, 0.4, 0.1, 0.15])
        self.check_buttons = CheckButtons(self.check_ax, [f'Band {i+1}' for i in range(self.bands)],
                                           self.selected_bands)
        self.check_buttons.on_clicked(self._toggle_bands)

        # Create slider for selecting brush size
        self.slider_ax = plt.axes([0.05, 0.2, 0.1, 0.1])
        self.slider = Slider(self.slider_ax, 'Brush Size', 0, 20, valinit=self.brush_size, valstep=1)
        self.slider.on_changed(self._update_brush_size)

        # Create button for saving the mask
        self.save_button_ax = plt.axes([0.05, 0.05, 0.1, 0.1])
        self.save_button = Button(self.save_button_ax, 'Save Mask')
        self.save_button.on_clicked(self.save_mask)

        # Connect mouse events to annotation functions
        self.fig.canvas.mpl_connect('button_press_event', self._start_annotate)
        self.fig.canvas.mpl_connect('motion_notify_event', self._drag_annotate)

        

        self._show_image()

        plt.show()

    def _toggle_bands(self, label):
        index = int(label.split()[1]) - 1
        self.selected_bands[index] = not self.selected_bands[index]
        self._show_image()

    def _update_brush_size(self, val):
        self.brush_size = int(val)

    def _show_image(self):
        self.ax.clear()
        self.ax.imshow(self._read_image(), cmap='gist_ncar')
        if self.annotated_mask is not None:
            self.ax.imshow(self.annotated_mask, cmap='binary', alpha=0.1)
        plt.draw()

    def _read_image(self):
        selected_indices = np.where(self.selected_bands)[0]
        image = np.dstack([self.dataset.read(i+1) for i in selected_indices.tolist()])
        if image.shape[2]==3:
            return image/np.max(image)
        return np.max(image, axis=2) if image.shape[2] > 1 else image.squeeze()

    def _start_annotate(self, event):
        if event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                x = int(event.xdata)
                y = int(event.ydata)
                self._annotate(x, y)

    def _drag_annotate(self, event):
        if event.xdata is not None and event.ydata is not None and event.button == 1 and event.inaxes == self.ax:
            x = int(event.xdata)
            y = int(event.ydata)
            self._annotate(x, y)

    def _annotate(self, x, y):
        if self.annotated_mask is None:
            self.annotated_mask = np.zeros_like(self.dataset.read(1), dtype=np.uint8)

        r, c = np.indices((self.brush_size * 1 , self.brush_size * 1 )) - self.brush_size
        valid_mask = (x + r >= 0) & (x + r < self.annotated_mask.shape[1]) & (y + c >= 0) & (y + c < self.annotated_mask.shape[0])
        with self.update_lock:
            self.annotated_mask[y + r[valid_mask], x + c[valid_mask]] = 255  # Annotation
        self._show_image()

    def save_mask(self, event):
        root = Tk()
        root.withdraw()  # Hide the main window
        output_file = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif")])
        root.destroy()  # Destroy the Tkinter window after file dialog is closed

        if output_file:
            if self.annotated_mask is not None:
                profile = self.dataset.profile
                profile.update(dtype=rasterio.uint8, count=1)

                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(self.annotated_mask, 1)
                    print(f"Mask saved to: {output_file}")
        plt.close()


parser=argparse.ArgumentParser(
          prog='Annotator',
          description='A tool for Annotating raster files, mainly .tiff'
                      'orthomosaics, returning a mask.tiff file which in combination with the original image.'
                      'can be used for getting refrene pixels and ROC analysis in the colorbased image segmentation scheme',
          epilog='Program written by Søren Vad Iversen (soiv@mmmi.sdu.dk) in '
                 '2024 as part of the Precisionseedbreeding project supported '
                 'by GUDP and Frøafgiftsfonden.')
parser.add_argument('reference_image', 
                    help='Path to the .tiff file that you want to Annotate.')
#parser.add_argument(--)

if __name__ == "__main__":
    args=parser.parse_args()
    raster_file = args.reference_image # Replace with your raster file path
    annotator = RasterAnnotator(raster_file)
