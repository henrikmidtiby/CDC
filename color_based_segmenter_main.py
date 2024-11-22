import argparse
import pathlib
import time

from tqdm import tqdm

from color_based_segmenter import ColorBasedSegmenter
from color_models import GaussianMixtureModelDistance, MahalanobisDistance
from orthomosaic_tiler import OrthomosaicTiles


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ColorDistranceCalculatorForOrthomosaics",
        description="A tool for calculating color distances in an "
        "orthomosaic to a reference color based on samples from "
        "an annotated image.",
        epilog="Program written by Henrik Skov Midtiby (hemi@mmmi.sdu.dk) in "
        "2023 as part of the Precisionseedbreeding project supported "
        "by GUDP and Frøafgiftsfonden.",
    )
    parser.add_argument("orthomosaic", help="Path to the orthomosaic that you want to process.", type=pathlib.Path)
    parser.add_argument("reference", help="Path to the reference image.", type=pathlib.Path)
    parser.add_argument("annotated", help="Path to the annotated reference image.", type=pathlib.Path)
    parser.add_argument(
        "--bands_to_use",
        default=None,
        type=int,
        nargs="+",
        help="The bands needed to be analysed, written as a lis, 0 indexed. If no value is specified all bands except alpha channel will be analysed.",
    )
    parser.add_argument(
        "--scale",
        default=5,
        type=float,
        help="The calculated distances are multiplied with this factor before the result is saved as an image. Default value is 5.",
    )
    parser.add_argument(
        "--output_tile_location",
        default="output/tiles",
        help="The location in which to save the tiles.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--mask_file_name",
        default="pixel_values",
        type=pathlib.Path,
        help="Change the name in which the pixel mask is saved. It defaults to pixel_values (.csv is automatically added)",
    )
    parser.add_argument(
        "--method",
        default="mahalanobis",
        help="The method used for calculating distances from the set of annotated pixels. Possible values are 'mahalanobis' for using the Mahalanobis distance and 'gmm' for using a Gaussian Mixture Model. 'mahalanobis' is the default value.",
    )
    parser.add_argument(
        "--param",
        default=2,
        type=int,
        help="Numerical parameter for the color model. When using the 'gmm' method, this equals the number of components in the Gaussian Mixture Model.",
    )
    parser.add_argument(
        "--tile_size",
        default=3000,
        type=int,
        help="The height and width of tiles that are analyzed. " "Default is 3000.",
    )
    parser.add_argument(
        "--run_specific_tile",
        nargs="+",
        type=int,
        metavar="TILE_ID",
        help="If set, only run the specific tile numbers. (--run_specific_tile 16 65) will run tile 16 and 65.",
    )
    parser.add_argument(
        "--run_specific_tileset",
        nargs="+",
        type=int,
        metavar="FROM_TILE_ID TO_TILE_ID",
        help="takes two inputs like (--from_specific_tileset 16 65). This will run every tile from 16 to 65.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ortho_tiler = OrthomosaicTiles(**vars(args))
    tile_list = ortho_tiler.divide_orthomosaic_into_tiles()

    if args.method == "mahalanobis":
        color_model = MahalanobisDistance(**vars(args))
    if args.method == "gmm":
        color_model = GaussianMixtureModelDistance(n_components=args.param, **vars(args))
    color_model.initialize()
    pixels_filename = args.output_tile_location.joinpath(f"{args.mask_file_name}.csv")
    color_model.save_pixel_values(pixels_filename)

    cbs = ColorBasedSegmenter(color_model=color_model, **vars(args))

    start = time.time()
    for tile in tqdm(tile_list):
        img, _ = tile.read_tile(args.orthomosaic, args.bands_to_use)
        distance_img = cbs.process_image(img)
        # tile.save_tile(distance_img, args.output_tile_location)
        tile.output = distance_img
    print("Time to run all tiles: ", time.time() - start)
    cbs.calculate_statistics(tile_list)
    cbs.save_statistics(args)

    output_filename = args.output_tile_location.joinpath("vegetation_test.tiff")
    ortho_tiler.save_orthomosaic_from_tile_output(output_filename)


if __name__ == "__main__":
    main()
