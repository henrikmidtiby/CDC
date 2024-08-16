# %%
#import sys
#sys.path.append('/home/soiv/Dropbox/VScode/Research assistant SDU/Gitlab_SDU/color-based-imagesegmentation/Segmenter')
import ROC.ROC as ROC
import rasterio


# %%
def get_img_from_tiff(path):
    with rasterio.open(path) as src:
     im=src.read()
    return im

# %%
path_to_img='/home/soiv/Dropbox/VScode/Research assistant SDU/Gitlab_SDU/Data/ROC_cutout_møn_mark3.tif'
path_to_segmented_img='/home/soiv/Dropbox/VScode/Research assistant SDU/Gitlab_SDU/Data/seg2_møn_mark3.tiff'
path_to_true_negative='/home/soiv/Dropbox/VScode/Research assistant SDU/Gitlab_SDU/Data/ROC_negative_mask_møn_mark3.tif'
path_to_true_positive='/home/soiv/Dropbox/VScode/Research assistant SDU/Gitlab_SDU/Data/ROC_positive_mask_møn_mark3.tif'
segmented_img= get_img_from_tiff(path_to_segmented_img)
true_negative=get_img_from_tiff(path_to_true_negative)
true_positive=get_img_from_tiff(path_to_true_positive)

# %%
with rasterio.open(path_to_img) as src:
    print( src.width )
    print( src.height )


roc=ROC.ROC()

# %%
#roc.get_points(segmented_img,true_positive,true_negative,100)
roc.distance_to_points(segmented_img,true_positive,true_negative)
roc.calculate_rates()
area=roc.calculate_area_under_graph()
print(area)
roc.plot_ROC()#options='precission')






