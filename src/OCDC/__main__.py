"""
CLI for running Color Segmentation on an Orthomosaic.

See ``OCDC --help`` for a list of arguments.
"""

from __future__ import annotations

import argparse
import pathlib
from datetime import datetime
from typing import Any

from OCDC.color_models import BaseDistance, GaussianMixtureModelDistance, MahalanobisDistance
from OCDC.tiled_color_based_distance import TiledColorBasedDistance
from OCDC.transforms import BaseTransform, GammaTransform, LambdaTransform


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="OCDC",
        description="""A tool for calculating color distances in an
                       orthomosaic to a reference color based on
                       samples from an annotated image.""",
        epilog=f"""Program written by SDU UAS Center (hemi@mmmi.sdu.dk) in
                   2023-{datetime.now().year} as part of the Precisionseedbreeding project
                   supported by GUDP and Frøafgiftsfonden.""",
    )
    parser.add_argument("orthomosaic", help="Path to the orthomosaic that you want to process.", type=pathlib.Path)
    parser.add_argument("reference", help="Path to the reference image.", type=pathlib.Path)
    parser.add_argument("annotated", help="Path to the annotated reference image.", type=pathlib.Path)
    parser.add_argument(
        "--bands_to_use",
        default=None,
        type=int,
        nargs="+",
        help="The bands needed to be analyzed, written as a list, 0 indexed. If no value is specified all bands except alpha channel will be analyzed.",
    )
    parser.add_argument(
        "--alpha_channel",
        default=-1,
        type=int,
        help="Alpha channel number 0 indexed. If no value is specified the last channel is assumed to be the alpha channel. If the orthomosaic does not contain an alpha channel use None.",
    )
    parser.add_argument(
        "--scale",
        default=5,
        type=float,
        help="The calculated distances are multiplied with this factor before the result is saved as an image. Default value is 5.",
    )
    parser.add_argument(
        "--output_location",
        default="output",
        help="The location in which to save the tiles and orthomosaic.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--do_not_save_orthomosaic",
        action="store_false",
        help="If set the no orthomosaic of the result is saved at output_location/orthomosaic.tiff. Default is to save orthomosaic.",
    )
    parser.add_argument(
        "--save_tiles",
        action="store_true",
        help="If set tiles are saved at output_location/tiles. Useful for debugging or parameter tweaking. Default no tiles are saved.",
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
        type=str,
        choices=["mahalanobis", "gmm"],
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
    return parser


def _parse_args(args: Any = None) -> Any:
    parser = _get_parser()
    return parser.parse_args(args)


def _process_transform_args(args: Any) -> dict[str, BaseTransform | None]:
    transform: BaseTransform | None = None
    if args.gamma_transform is not None:
        transform = GammaTransform(args.gamma_transform)
    if args.lambda_transform is not None:
        transform = LambdaTransform(args.lambda_transform)
    return {"transform": transform}


def _process_color_model_args(args: Any, keyword_args: dict[str, Any], save_pixels_values: bool = True) -> BaseDistance:
    if args.method == "mahalanobis":
        color_model: BaseDistance = MahalanobisDistance(**keyword_args)
    elif args.method == "gmm":
        color_model = GaussianMixtureModelDistance(n_components=args.param, **keyword_args)
    else:
        raise ValueError(f"Method must be one of 'mahalanobis' or 'gmm', but got {args.method}")
    if save_pixels_values:
        pixels_filename = args.output_location.joinpath(f"{args.mask_file_name}.csv")
        color_model.save_pixel_values(pixels_filename)
    return color_model


def _main() -> None:
    args = _parse_args()
    print(args)
    keyword_args = vars(args)
    keyword_args.update(_process_transform_args(args))
    color_model = _process_color_model_args(args, keyword_args)
    tcbs = TiledColorBasedDistance(color_model=color_model, **keyword_args)
    tcbs.process_tiles(save_tiles=args.save_tiles, save_ortho=args.do_not_save_orthomosaic)
    tcbs.save_statistics(args)


if __name__ == "__main__":
    _main()
