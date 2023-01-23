import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import numpy as np
import cv2
from shapely.geometry import shape, box
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# add openslide
OPENSLIDE_PATH = r'c://users/koenk/documents/EMC-Seminoma-Relapse/openslide-win64/bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process commandline arguments.")

    parser.add_argument('-i', '--input', type=dir_path,
                        help="Path to whole slide images and annotations.",
                        required=True)
    parser.add_argument('-o', '--output', type=tar_path, nargs="?", const=1,
                        help="Path to output folder.",
                        required=True)
    parser.add_argument('-s', '--tilesize', type=int, nargs="?", const=1,
                        default=224,
                        help="Size (px) of the square tiles to be extracted.")
    parser.add_argument('-m', '--magnification', type=int, nargs="?", const=1,
                        default=20,
                        help="Magnification used in the extracted tiles.")

    return parser.parse_args()


def dir_path(path):
    if os.path.isdir(path):
        return path
    raise NotADirectoryError(path)


def tar_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"Output folder {path} created")
        return path
    return path


def tile_slide(annotation_path, tile_size, downsample):
    # load annotation(s)
    with open(annotation_path, 'r', encoding="utf8") as file:
        annotation = json.load(file)

    annotation_polys = [shape(f["geometry"]) for f in annotation["features"]]

    # initialize bounding box around first annotation
    bbox = annotation_polys[0].bounds

    if len(annotation_polys) > 1:
        # iterate over annotations, if annotation found to be outside bounding box,
        # adjust edge to that annotation
        for poly in annotation_polys:
            bbox = (min(bbox[0], poly.bounds[0]), min(bbox[1], poly.bounds[1]),
                    max(bbox[2], poly.bounds[2]), max(bbox[3], poly.bounds[3]))

    # bbox is stored as (xmin, ymin, xmax, ymax)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # calculate the width and difference with the old so that the box becomes a multiple
    # of the tile size
    new_width, width_diff = np.ceil(width / tile_size[0]) * tile_size[0], \
                            np.ceil(width / tile_size[0]) * tile_size[0] - width
    new_height, height_diff = np.ceil(height / tile_size[1]) * tile_size[1], \
                              np.ceil(height / tile_size[1]) * tile_size[1] - height

    # shift box so that it is centered around all annotations
    bbox_scaled = (
        bbox[0] - width_diff // 2,
        bbox[1] - height_diff // 2,
        bbox[0] + new_width - width_diff // 2,
        bbox[1] + new_height - height_diff // 2
    )

    # get x and y gridpoints the tiles should be generated on
    x_range = np.arange(bbox_scaled[0], bbox_scaled[2], tile_size[0] * downsample)
    y_range = np.arange(bbox_scaled[1], bbox_scaled[3], tile_size[1] * downsample)

    # check the overlapping between the tile and the annotation polygons, discard tiles that
    # do not intersect with the annotations
    tile_coordinates = [(int(x), int(y)) for x, y in product(x_range, y_range)
                        if any(poly.contains(box(x, y, x + tile_size[0] * downsample, y + tile_size[1] * downsample))
                        for poly in annotation_polys)]
    count = len(tile_coordinates)

    return tile_coordinates, count


def process_and_save_tile(slide, coord, tile_size, output_path):
    offset_x = int(slide.properties['openslide.bounds-x'])
    offset_y = int(slide.properties['openslide.bounds-y'])
    tile = slide.read_region((offset_x + coord[0],offset_y + coord[1]), 2, (tile_size, tile_size))
    tile = np.asarray(tile)

    gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2GRAY)

    _, threshold_tile = cv2.threshold(gray_tile, 240, 255, cv2.THRESH_BINARY_INV)

    tissue_pixels = np.count_nonzero(threshold_tile)
    tissue_percentage = tissue_pixels / (threshold_tile.shape[0] * threshold_tile.shape[1]) * 100

    if tissue_percentage > 80:
        cv2.imwrite(os.path.join(output_path, f"{coord[0]}_{coord[1]}.png"), cv2.cvtColor(tile, cv2.COLOR_RGBA2BGRA))


def main():
    # Get input from commandline
    parsed_args = parse_arguments()

    base_dir = parsed_args.input

    image_annotation_paths = []
    for dirpath, dirnames, filenames in os.walk(os.path.abspath(os.path.join(base_dir, 'Converted and anonymized'))):
        for filename in filenames:
            if filename.endswith('.mrxs'):
                image_path = os.path.join(dirpath, filename)
                annotation_path = os.path.abspath(os.path.join(base_dir, 'Annotations', os.path.relpath(dirpath, os.path.abspath(os.path.join(base_dir, 'Converted and anonymized'))), filename.replace('.mrxs', '.geojson')))
                if os.path.exists(annotation_path):
                    image_annotation_paths.append((image_path, annotation_path))

    tile_size = (parsed_args.tilesize, parsed_args.tilesize)

    for slide_path, annotation_path in image_annotation_paths:
        output_path = os.path.join(parsed_args.output, os.path.basename(slide_path).replace('.mrxs', ''))
        print(output_path)

        slide = openslide.open_slide(slide_path)

        objective_power = slide.properties['openslide.objective-power']
        downsample = int(int(objective_power) / parsed_args.magnification)

        tiles_coords, num_tiles = tile_slide(annotation_path, tile_size, downsample)

        # FOR TESTING PURPOSES:
        # tiles_coords = tiles_coords[:25]

        print(f'Processing slide {slide_path}: generating {num_tiles} tiles.')

        # Uncomment this code to use multiple threads to process WSI

        tar_path(output_path)

        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda coord: process_and_save_tile(slide, coord, tile_size[0], output_path), tiles_coords)))

        print('----' * 8)


if __name__ == "__main__":
    main()
