001_get_cutout_from_orthomosaic: gdal_translate -projwin 322761.1344 6088831.9431 322765.846 6088829.0647 -of GTiff /home/soiv/Downloads/Orthomosaic-møn-mark3.tif "/home/soiv/Dropbox/VScode/Research assistant SDU/Gitlab_SDU/Data/Reference_møn_mark3.tif"

003_segment_small_image: python3 segmenter.py '/home/soiv/Dropbox/VScode/Research assistant SDU/Gitlab_SDU/Data/ROC_cutout_møn_mark3.tif' '/home/soiv/Dropbox/VScode/Research assistant SDU/Gitlab_SDU/Data/Reference_møn_mark3.tif' '/home/soiv/Dropbox/VScode/Research assistant SDU/Gitlab_SDU/Data/reference_mask_møn_mark3.tif' --notiling

