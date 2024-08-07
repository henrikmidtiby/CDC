#For adding functionality to make transformation between different colorspaces 









def transform_image(img,transform=None,):
    if transform== None:
        return img
    
    transformed_img=transform(img)
    return transformed_img