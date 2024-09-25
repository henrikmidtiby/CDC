
import numpy as np
import rasterio
from sklearn import mixture
import cv2
from icecream import ic





#Structure for gathering information needed for colormodels


class ReferencePixels:
    def __init__(self):
        self.reference_image = None
        self.mask = None
        self.values = None
        self.bands_to_use=None
    def load_reference_image(self, filename_reference_image):
        with rasterio.open(filename_reference_image) as ref_img:
            self.reference_image =ref_img.read()

    def load_mask(self, filename_mask):
        with rasterio.open(filename_mask) as msk:
            self.mask=msk.read()

    def generate_pixels_from_mask(self,bands_to_use):
        

        number_of_bands=self.reference_image.shape[0]-1
        
        if bands_to_use == None:
            self.bands_to_use=list(range(number_of_bands))
        else:
            self.bands_to_use = bands_to_use
        if len(self.bands_to_use) > number_of_bands:
            raise Exception("Chosen number of bands larger than number fo bands in sample")
        for i in self.bands_to_use:
            if i > number_of_bands:
                raise Exception("no bands of the chosen index")
        
        pixels=np.reshape(self.reference_image[self.bands_to_use,:,:],(len(self.bands_to_use),-1))
        pixels = pixels.transpose()
        mask_pixels = np.reshape(self.mask, (-1))
        self.values = pixels[mask_pixels >= 200,].transpose()

    def show_statistics_of_pixel_mask(self):
        print(f"Number of annotated pixels: { self.values.shape }")
        if self.values.shape[1] < 100:
            raise Exception("Not enough annotated pixels")

    def save_pixel_values_to_file(self, filename):
        print(f"Writing pixel values to the file \"{ filename }\"")
        np.savetxt(filename, 
                   self.values.transpose(), 
                   delimiter = '\t', 
                   fmt='%i', 
                   header="b\tg\tr", 
                   comments = "")



class ReferencePixels_jpg:
    def __init__(self):
        self.reference_image = None
        self.annotated_image = None
        self.pixel_mask = None
        self.bands_to_use=None
        self.values = None
        #self.colorspace = colorspace()

    def load_reference_image(self, filename_reference_image):
        self.reference_image = cv2.imread(filename_reference_image)
        
    def load_annotated_image(self, filename_annotated_image):
        self.annotated_image = cv2.imread(filename_annotated_image)
        
    def generate_pixel_mask(self,bands_to_use,
                            lower_range=(0, 0, 245),
                            higher_range=(10, 10, 256)):

    
        if bands_to_use is None:
            self.bands_to_use=list(range(3))
        else:
            self.bands_to_use = bands_to_use
        

        self.pixel_mask = cv2.inRange(self.annotated_image,
                                      lower_range,
                                      higher_range)
        pixels = np.reshape(self.reference_image[:,:,self.bands_to_use],(-1,len(self.bands_to_use)))
        mask_pixels = np.reshape(self.pixel_mask, (-1))
        self.values = pixels[mask_pixels == 255, ].transpose()

    def show_statistics_of_pixel_mask(self):
        print(f"Number of annotated pixels: { self.values.shape }")
        if self.values.shape[1] < 100:
            raise Exception("Not enough annotated pixels")

    def save_pixel_values_to_file(self, filename):
        print(f"Writing pixel values to the file \"{ filename }\"")
        np.savetxt(filename, 
                   self.values.transpose(), 
                   delimiter = '\t', 
                   fmt='%i', 
                   header="b\tg\tr", 
                   comments = "")




def get_referencepixels(ref_img,mask,bands_to_use,filename,method=None):
    match method:
        case None :
            reference_pixels=ReferencePixels()
            reference_pixels.load_reference_image(ref_img)
            reference_pixels.load_mask(mask)
            reference_pixels.generate_pixels_from_mask(bands_to_use)
            reference_pixels.save_pixel_values_to_file(filename)
        case 'jpg':
            reference_pixels=ReferencePixels_jpg()
            reference_pixels.load_reference_image(ref_img)
            reference_pixels.load_annotated_image(mask)
            reference_pixels.generate_pixel_mask(bands_to_use)
            reference_pixels.save_pixel_values_to_file(filename)
        case _:
            raise Exception(print('Not viable method for generating Reference pixels'))
        
    return reference_pixels
    



#-----------------------------------------------------------------------------------------------------------
#colormodels:



class MahalanobisDistance:
    """
    A multivariate normal distribution used to describe the color of a set of
    pixels.
    """
    def __init__(self):
        self.average = None
        self.covariance = None
        self.bands_to_use=None
        

    def calculate_statistics(self, reference_pixels,transform):
        print(reference_pixels)
        transformed_reference_pixels = transform.transform_image( reference_pixels )


        self.covariance = np.cov(transformed_reference_pixels)
        self.average = np.average(transformed_reference_pixels, axis=1)
       
        if self.covariance.shape is ():
            self.covariance=np.reshape(self.covariance,(1,1))

        

    def calculate_distance(self, image,transform):
        """
        For all pixels in the image, calculate the Mahalanobis distance
        to the reference color.
        """
        
        pixels = np.reshape(image[self.bands_to_use,:,:], (len(self.bands_to_use),-1))
        transformed_pixels=transform.transform_image(pixels).transpose()


        inv_cov = np.linalg.inv(self.covariance)

        
        diff = transformed_pixels - self.average
        modified_dot_product = diff * (diff @ inv_cov)
        distance = np.sum(modified_dot_product, axis=1)
        distance = np.sqrt(distance)

        distance_image = np.reshape(distance, (1,image.shape[1], image.shape[2]))

        return distance_image
    




    def show_statistics(self):
        print("Average color value of annotated pixels")
        print(self.average)
        print("Covariance matrix of the annotated pixels")
        print(self.covariance)

class GaussianMixtureModelDistance:
    def __init__(self, n_components):
        self.gmm = None
        self.n_components = n_components
        self.bands_to_use=None
        self.globalmin=None

        self.minimum_is_not_at_mean=False

    def calculate_statistics(self, reference_pixels,transform):
        self.gmm = mixture.GaussianMixture(n_components=self.n_components,
                                           covariance_type="full")
        transformed_reference_pixels=transform.transform_image(reference_pixels).transpose()
        self.gmm.fit(transformed_reference_pixels)

        self.globalmin=np.amin( self.gmm.score_samples(self.gmm.means_ ))
        

    def calculate_distance(self, image,transform):
        """
        For all pixels in the image, calculate the distance to the
        reference color modelled as a Gaussian Mixture Model.
        """
        pixels = np.reshape(image[self.bands_to_use,:,:], (len(self.bands_to_use),-1))
        transformed_pixels=transform.transform_image(pixels).transpose()
        loglikelihood=self.gmm.score_samples(transformed_pixels)
     
        
        distance=self.log_likelihood_to_distance(loglikelihood)

        distance_image = np.reshape(distance, (1,image.shape[1], image.shape[2]))
        
        return distance_image

    def log_likelihood_to_distance(self,loglikelihood):
        """Function for converting the gmm loglikelyhood to such that it is only positive values"""
        distance = -(loglikelihood-self.globalmin)
        
        
        if np.any(distance<0):
            if not self.minimum_is_not_at_mean:
                print("Warning: the global minimum of the -log(Likelyhood) is not at the center of either of the clusters.")
        
        distance[distance<0]=0 #sets any still negative value to zero 

        return distance
    


    
    def show_statistics(self):
        print("GMM")
        print(self.gmm)
        print(f'GMM means= {self.gmm.means_}')
        print(f'GMM covariance= {self.gmm.covariances_}')










def initialize_colormodel(reference_pixels,method,param,transform):
    model= None
    match method:
        case 'mahalanobis':
            model=MahalanobisDistance()
        case 'gmm':
            model=GaussianMixtureModelDistance(param)
        case _:
            print("The method selected does not match any known colormodel methods, Mahalanobis Distance was used instead")
            model=MahalanobisDistance()
    
    model.bands_to_use=reference_pixels.bands_to_use
    model.calculate_statistics(reference_pixels.values,transform)
    model.show_statistics()
    return model




#--------------------------------------------------------------------------------------------------------------
















































