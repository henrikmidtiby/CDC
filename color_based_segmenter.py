import argparse
import pathlib
from typing import Any

from color_models import BaseDistance, GaussianMixtureModelDistance, MahalanobisDistance
from tiled_color_based_segmenter import TiledColorBasedSegmenter
from transforms import BaseTransformer, GammaTransform, LambdaTransform


def parse_args() -> Any:
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
        help="The bands needed to be analysed, written as a list, 0 indexed. If no value is specified all bands except alpha channel will be analysed.",
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
    parser.add_argument(
        "--gamma_transform",
        type=float,
        default=None,
        metavar="GAMMA",
        help="Apply a gamma transform with the given gamma to all inputs. Default no transform.",
    )
    parser.add_argument(
        "--lambda_transform",
        type=str,
        default=None,
        metavar="LAMBDA",
        help="Apply a Lambda transform with the given Lambda expression to all inputs. Numpy is available as np. Default no transform.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    keyword_args = vars(args)
    transform: BaseTransformer | None = None
    if args.gamma_transform is not None:
        transform = GammaTransform(args.gamma_transform)
    if args.lambda_transform is not None:
        transform = LambdaTransform(args.lambda_transform)
    keyword_args.update({"transform": transform})
    if args.method == "mahalanobis":
        color_model: BaseDistance = MahalanobisDistance(**keyword_args)
    if args.method == "gmm":
        color_model = GaussianMixtureModelDistance(n_components=args.param, **keyword_args)
    pixels_filename = args.output_tile_location.joinpath(f"{args.mask_file_name}.csv")
    color_model.save_pixel_values(pixels_filename)

    tcbs = TiledColorBasedSegmenter(color_model=color_model, **keyword_args)
    tcbs.process_tiles()
    tcbs.calculate_statistics()
    tcbs.save_statistics(args)


if __name__ == "__main__":
    main()
