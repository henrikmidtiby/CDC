"""
CLI for running Color Segmentation on an Orthomosaic.

See ``CDC --help`` for a list of arguments.
"""

from __future__ import annotations

import argparse
import os
import pathlib
from datetime import datetime
from typing import Any

from CDC.color_models import BaseDistance, GaussianMixtureModelDistance, MahalanobisDistance
from CDC.orthomosaic_tiler import OrthomosaicTiles
from CDC.tiled_color_based_distance import TiledColorBasedDistance
from CDC.transforms import BaseTransform, GammaTransform, LambdaTransform


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="CDC",
        description="""A tool for calculating color distances in an
                       orthomosaic to a reference color based on
                       samples from an annotated image.""",
        epilog=f"""Program written by SDU UAS Center (hemi@mmmi.sdu.dk) in
                   2023-{datetime.now().year} as part of the Precisionseedbreeding project
                   supported by GUDP and Frøafgiftsfonden.""",
    )
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("orthomosaic", help="Path to the orthomosaic that you want to process.", type=pathlib.Path)
    input_group.add_argument("reference", help="Path to the reference image.", type=pathlib.Path)
    input_group.add_argument("annotated", help="Path to the annotated reference image.", type=pathlib.Path)
    # input_group.add_argument(
    #     "--ref_pixels_from_csv",
    #     type=pathlib.Path,
    #     default=None,
    #     metavar="CSV",
    #     help="Load reference pixel from csv file instead of from reference and annotated image.",
    # )
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_location",
        default="output",
        help="The location in which to save the tiles and orthomosaic.",
        type=pathlib.Path,
    )
    output_group.add_argument(
        "--do_not_save_orthomosaic",
        action="store_false",
        help="If set the no orthomosaic of the result is saved at output_location/orthomosaic.tiff. Default is to save orthomosaic.",
    )
    output_group.add_argument(
        "--save_tiles",
        action="store_true",
        help="If set tiles are saved at output_location/tiles. Useful for debugging or parameter tweaking. Default no tiles are saved.",
    )
    output_group.add_argument(
        "--save_ref_pixels",
        action="store_true",
        help="Save the raw, transformed and selected reference pixels in output_location/pixel_values. Default do not save.",
    )
    output_group.add_argument(
        "--save_statistics",
        action="store_true",
        help="Save statistics of the processed orthomosaic including a histogram of color distances. Files are saved in output_location/statistics. Default do not save.",
    )
    color_model_group = parser.add_argument_group("Color Model")
    color_model_group.add_argument(
        "--method",
        default="mahalanobis",
        type=str,
        choices=["mahalanobis", "gmm"],
        help="The method used for calculating distances from the set of annotated pixels. Possible values are 'mahalanobis' for using the Mahalanobis distance and 'gmm' for using a Gaussian Mixture Model. 'mahalanobis' is the default value.",
    )
    color_model_group.add_argument(
        "--param",
        default=2,
        type=int,
        help="Numerical parameter for the color model. When using the 'gmm' method, this equals the number of components in the Gaussian Mixture Model.",
    )
    color_model_group.add_argument(
        "--scale",
        default=5,
        type=float,
        help="The calculated distances are multiplied with this factor before the result is saved as an image. Default value is 5.",
    )
    transform_group = parser.add_argument_group("Transforms")
    transform_group.add_argument(
        "--gamma_transform",
        type=float,
        default=None,
        metavar="GAMMA",
        help="Apply a gamma transform with the given gamma to all inputs. Default no transform.",
    )
    transform_group.add_argument(
        "--lambda_transform",
        type=str,
        default=None,
        metavar="LAMBDA",
        help="Apply a Lambda transform with the given Lambda expression to all inputs. Numpy is available as np. Default no transform.",
    )
    image_channels_group = parser.add_argument_group("Image Channels")
    image_channels_group.add_argument(
        "--bands_to_use",
        default=None,
        type=int,
        nargs="+",
        help="The bands needed to be analyzed, written as a list, 0 indexed. If no value is specified all bands except alpha channel will be analyzed.",
    )
    image_channels_group.add_argument(
        "--alpha_channel",
        default=-1,
        type=int,
        help="Alpha channel number 0 indexed. If no value is specified the last channel is assumed to be the alpha channel. If the orthomosaic does not contain an alpha channel use None.",
    )
    image_channels_group.add_argument(
        "--channel_names_in",
        default=None,
        type=str,
        help="Names of the image channels in the input image separated by comma, i.e. (R, G, B, A). Default channels are named c1, c2, ...",
    )
    image_channels_group.add_argument(
        "--channel_names_out",
        default=None,
        type=str,
        help="Names of the image channels after the transform separated by comma, i.e. (H, S, V). Default channels are named c1, c2, ...",
    )
    tile_group = parser.add_argument_group("Tiles")
    tile_group.add_argument(
        "--tile_size",
        default=3000,
        type=int,
        help="The height and width of tiles that are analyzed. " "Default is 3000.",
    )
    tile_group.add_argument(
        "--run_specific_tile",
        nargs="+",
        type=int,
        metavar="TILE_ID",
        help="If set, only run the specific tile numbers. (--run_specific_tile 16 65) will run tile 16 and 65.",
    )
    tile_group.add_argument(
        "--run_specific_tileset",
        nargs="+",
        type=int,
        metavar="FROM_TILE_ID TO_TILE_ID",
        help="takes two inputs like (--from_specific_tileset 16 65). This will run every tile from 16 to 65.",
    )
    return parser


def _parse_args(args: Any = None) -> Any:
    parser = _get_parser()
    return parser.parse_args(args)


def _create_output_location(output_directory: pathlib.Path) -> None:
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)


def _process_transform_args(args: Any) -> dict[str, BaseTransform | None]:
    transform: BaseTransform | None = None
    if args.gamma_transform is not None:
        transform = GammaTransform(args.gamma_transform)
    if args.lambda_transform is not None:
        transform = LambdaTransform(args.lambda_transform)
    return {"transform": transform}


def _process_color_model_args(args: Any, keyword_args: dict[str, Any]) -> BaseDistance:
    color_args = {
        "reference": keyword_args["reference"],
        "annotated": keyword_args["annotated"],
        "bands_to_use": keyword_args["bands_to_use"],
        "alpha_channel": keyword_args["alpha_channel"],
        "transform": keyword_args["transform"],
    }
    if args.method == "mahalanobis":
        color_model: BaseDistance = MahalanobisDistance.from_image_annotation(**color_args)
    elif args.method == "gmm":
        color_model = GaussianMixtureModelDistance.from_image_annotation(n_components=args.param, **color_args)
    else:
        raise ValueError(f"Method must be one of 'mahalanobis' or 'gmm', but got {args.method}")
    if args.save_ref_pixels:
        color_model.save_pixel_values(args.output_location, args.channel_names_in, args.channel_names_out)
    return color_model


def _main() -> None:
    args = _parse_args()
    keyword_args = vars(args)
    keyword_args.update(_process_transform_args(args))
    _create_output_location(args.output_location)
    try:
        color_model = _process_color_model_args(args, keyword_args)
        ortho_tiler = OrthomosaicTiles(
            orthomosaic=args.orthomosaic,
            tile_size=args.tile_size,
            run_specific_tile=args.run_specific_tile,
            run_specific_tileset=args.run_specific_tileset,
        )
        tcbs = TiledColorBasedDistance(
            ortho_tiler=ortho_tiler, color_model=color_model, scale=args.scale, output_location=args.output_location
        )
        tcbs.process_tiles(save_tiles=args.save_tiles, save_ortho=args.do_not_save_orthomosaic)
        if args.save_statistics:
            tcbs.save_statistics(args)
    except Exception as e:
        print(f"{ type(e).__name__}: {e}")


if __name__ == "__main__":
    _main()
