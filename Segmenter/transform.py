#For adding functionality to make transformation between different colorspaces 
class Transformer:
    def __init__(self):
        self.transform=None


    def define_transform(self, func):

        self.transform=func

        return 1


    def transform_image(self,img):
        if self.transform== None:
            return img
    
        transformed_img=self.transform(img)
        return transformed_img
    



