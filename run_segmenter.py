import Segmenter.segmenter as segmenter
import Segmenter.tiler as tiler
import Segmenter.colormodels as colormodels
import argparse


def segment_orthomosaic(args):    
    referencepixels=colormodels.get_referencepixels(args.reference,args.mask,args.bands_to_use,args.ref_pixel_filename,args.ref_method)
    colormodel=colormodels.initialize_colormodel(referencepixels,args.method,args.param)
    cbs = segmenter.ColorBasedSegmenter()
    cbs.initialize_segmenter(args.output_tile_location,colormodel,args.scale)
    if args.notiling==False:
        tile_list=tiler.get_tilelist(args.orthomosaic,int(args.tile_size))
           
        cbs.apply_colormodel_to_tiles(tile_list)
    if args.notiling==True:
        single_tile=tiler.get_single_tile(args.orthomosaic)
        cbs.apply_colormodel_to_single_tile(single_tile)
        

    

    







if __name__=='__main__':
#%%#
    parser = argparse.ArgumentParser(
            prog='ColorDistranceCalculatorForOrthomosaics',
            description='A tool for calculating color distances in an '
                        'orthomosaic to a reference color based on samples from '
                        'an annotated image.',
            epilog='Program written by Henrik Skov Midtiby (hemi@mmmi.sdu.dk) in '
                    '2023 as part of the Precisionseedbreeding project supported '
                    'by GUDP and Frøafgiftsfonden.')
    parser.add_argument('orthomosaic', 
                        help='Path to the orthomosaic that you want to process.')
    parser.add_argument('reference', 
                        help='Path to the reference image.')
    parser.add_argument('mask', 
                        help='Path to the annotated reference image.')
    parser.add_argument('--bands_to_use',
                        default=None,
                        type=int,
                        nargs='+',
                        help='The bands needed to be analysed, written as a lis, 0 indexed'
                            'If no value is speciefied all bands except alpha channel will be analysed.')
    parser.add_argument('--ref_pixel_filename',
                        default='Referencepixels.txt',)
    parser.add_argument('--scale', 
                        default=5,
                        type=float,
                        help='The calculated distances are multiplied with this '
                            'factor before the result is saved as an image. '
                            'Default value is 5.')
    parser.add_argument('--tile_size',
                        default=3000,
                        help='The height and width of tiles that are analyzed. '
                            'Default is 3000.')
    parser.add_argument('--output_tile_location', 
                        default='output/mahal',
                        help='The location in which to save the mahalanobis tiles.')
    parser.add_argument('--input_tile_location', 
                        default=None,
                        help='The location in which to save the input tiles.')
    parser.add_argument('--method',
                        default='mahalanobis',
                        help='The method used for calculating distances from the '
                            'set of annotated pixels. '
                            'Possible values are \'mahalanobis\' for using the '
                            'Mahalanobis distance and '
                            '\'gmm\' for using a Gaussian Mixture Model.'
                            '\'mahalanobis\' is the default value.')
    parser.add_argument('--param',
                        default=2,
                        type=int,
                        help='Numerical parameter for the color model. '
                            'When using the \'gmm\' method, this equals the '
                            'number of components in the Gaussian Mixture Model.')
    parser.add_argument( '--notiling',
                        action="store_true",
                        help='Options for choosing not to seperate orthomosaic into tiles')
    parser.add_argument( '--ref_method',
                        default=None,
                        help='Method for generating Reference pixels, default is from Mask( .tiff file), other option is red anotated jpg file.')
    args = parser.parse_args()

    segment_orthomosaic(args)
