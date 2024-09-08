extract_images:
	gdal_translate -srcwin 10000 8000 3000 3000 -of PNG ~/Nextcloud/Shared/2023/2023-02-14\ ScoutRobotics\ data/Orthomosaic-møn-mark3.tif crop_from_orthomosaic.png

add_text_to_images:
	convert -resize 400% -font helvetica -fill white -pointsize 60 -gravity SouthWest -draw "text 10, 10 '1)'" crop_from_orthomosaic_101_109.png crop_from_orthomosaic_101_109_with_label.png
	convert -resize 400% -font helvetica -fill white -pointsize 60 -gravity SouthWest -draw "text 10, 10 '2)'" crop_from_orthomosaic_101_109-segmented.png crop_from_orthomosaic_101_109-segmented_with_label.png
	convert -resize 400% -font helvetica -fill white -pointsize 60 -gravity SouthWest -draw "text 10, 10 '3)'" crop_from_orthomosaic_101_109-annotated.png crop_from_orthomosaic_101_109-annotated_with_label.png
