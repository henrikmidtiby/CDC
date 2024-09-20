
import ROC.ROC as ROC
import rasterio
import argparse

def get_img_from_tiff(path):
    with rasterio.open(path) as src:
     im=src.read()
    return im

def run_ROC(path_to_distance_img,path_to_pos_mask,path_to_neg_mask,plot_options):
    distance_img= get_img_from_tiff(path_to_distance_img)
    true_positive=get_img_from_tiff(path_to_pos_mask)
    true_negative=get_img_from_tiff(path_to_neg_mask)
    roc=ROC.ROC()


    roc.distance_to_points(distance_img,true_positive,true_negative)
    roc.calculate_rates()
    area=roc.calculate_area_under_graph()
    roc.plot_ROC(plot_options)
    print(f"The Area under curve is approximately: {area}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            prog='RecieverOperatorCharacteristics',
            description='A tool for calculating Reciever Operator Characteristics for colorbased image segmentation.' 
                        'Based on distance image, positive mask and negative mask.',
            epilog='Program written by Søren Vad Iversen(soiv@mmmi.sdu.dk) in '
                    '2024 as part of the Precisionseedbreeding project supported '
                    'by GUDP and Frøafgiftsfonden.')
    parser.add_argument('path_to_distance_image',
                    help='The path for the distance image(.tif) for which the positive and negative mask has been made')
    parser.add_argument('path_to_positive_mask',
                    help='The path to the file(.tif) containing the a mask of the indicating places in the image where where the segmentation should characterise as positive, with value 255 indicating positive.')
    parser.add_argument('path_to_negative_mask',
                    help='The path to the file(.tif) containing the a mask of the indicating places in the image where where the segmentation should NOT characterise as positive, with value 255 indicating negative.')
    parser.add_argument('--plot_options',
                    default=None,
                    help='Plotting options default is False Positive Rate alternative plotting options are precision')
    

    args=parser.parse_args()


    run_ROC(args.path_to_distance_image,args.path_to_positive_mask,args.path_to_negative_mask,args.plot_options)