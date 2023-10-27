"""
Script used to generate tiles from WSIs (stored in .mrxs file format).

The script takes 4 arguments: input path (where WSIs and annotations are stored), output path
(where tiles are saved), tilesize (original region size) and magnification (at what level the 
tiles are extracted). Please note that tiles are always saved in size 224x224, regardless of the
tilesize set. This behaviour can be changed in the script itself.
Tiles that contain less than 80% tissue are automatically rejected and not saved.
The OPENSLIDE_PATH variable should be set to the bin folder in the openslide folder.
The script will save the tiles in folders with the structure: Patient ID --> Slide ID --> Tiles.
 
Author: Koen Kwakkenbos
(k.kwakkenbos@student.tudelft.nl/k.kwakkenbos@gmail.com)
Version: 1.0
Date: Feb 2022
"""

import os
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import numpy as np
import cv2
from shapely.geometry import shape, box
from tqdm import tqdm

from helpers import dir_path, tar_path

# Add openslide.
# ! ADD THIS PATH MANUALLY
OPENSLIDE_PATH = r'C:\Users\093637\Documents\EMC-Seminoma-Relapse\openslide-win64\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process commandline arguments.")

    parser.add_argument('-i', '--input', type=dir_path,
                        help="Path to whole slide images and annotations.",
                        required=True)
    parser.add_argument('-o', '--output', type=tar_path, nargs="?", const=1,
                        help="Path to output folder.",
                        required=True)
    parser.add_argument('-s', '--source', type=str, choices=["EMC", "TCGA"], nargs="?", const=1,
                        help="Data source.",
                        required=True)
    parser.add_argument('-ts', '--tilesize', type=int, nargs="?", const=1,
                        default=224,
                        help="Size (px) of the square tiles to be extracted.")
    parser.add_argument('-m', '--magnification', type=int, nargs="?", const=1,
                        default=20,
                        help="Magnification used in the extracted tiles.")

    return parser.parse_args()


def tile_slide(annotation_path, tile_size, downsample):
    """
    Generate coordinates for tiles that overlap with annotations in a whole slide image.

    Parameters:
    - annotation_path (str): The path to the annotation file in GeoJSON format.
    - tile_size (tuple): A tuple of the size of the tiles to be generated in pixels.
    - downsample (int): The amount to downsample the image by.

    Returns:
    - tuple: A tuple containing a list of tile coordinates that overlap with annotations, and the number of tiles.

    """
    # load annotation(s)
    with open(annotation_path, 'r', encoding="utf8") as file:
        annotation = json.load(file)

    # Check if the annotation has holes (and if so, only take the exterior of the annotation)
    try:
        annotation_polys = [poly for poly in list(shape(f["geometry"]).geoms) for f in annotation["features"]]
    except:
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

    # Calculate the overlap size based on the original tile size
    overlap_size = (int(tile_size[0] * 0.2), int(tile_size[1] * 0.2))

    # Adjust the tile size to account for overlap
    adjusted_tile_size = (tile_size[0] - overlap_size[0], tile_size[1] - overlap_size[1])

    # get x and y gridpoints the tiles should be generated on
    x_range = np.arange(bbox_scaled[0], bbox_scaled[2] - adjusted_tile_size[0] * downsample, 
                        adjusted_tile_size[0] * downsample)
    y_range = np.arange(bbox_scaled[1], bbox_scaled[3] - adjusted_tile_size[1] * downsample, 
                        adjusted_tile_size[1] * downsample)
    # check the overlapping between the tile and the annotation polygons, discard tiles that
    # do not intersect with the annotations
    
    tile_coordinates = []
    
    for x, y in product(x_range, y_range):
        tile_box = box(x, y, x + (tile_size[0] * downsample), y + (tile_size[1] * downsample))
        for poly in annotation_polys:
            if poly.intersects(tile_box):
                if poly.intersection(tile_box).area / tile_box.area > 0.8:
                    tile_coordinates.append((int(x), int(y)))
            
    # REMOVE THIS!!!
    # tile_coordinates = random.choices(tile_coordinates, k=200)
    # --------------

    count = len(tile_coordinates)

    return tile_coordinates, count


def process_and_save_tile(slide, coord, tile_size, downsample, output_path):
    """
    Given a slide image, the coordinates of a tile in the slide image, the size of the tile,
    and the output directory, extracts the tile, processes it, and saves it as a PNG file if
    it contains more than 80% tissue.

    Parameters:
    - slide (OpenSlide object): An OpenSlide object representing a whole slide image.
    - coord (tuple): A tuple of two integers representing the (x,y) coordinates of the top-left
      corner of the tile.
    - tile_size (int): Size of the tile(s) to generate in pixels.
    - output_path (str): The path to the directory where the generated tile should be saved.

    Returns:
    - None
    """

    try:
        offset_x = int(slide.properties['openslide.bounds-x'])
        offset_y = int(slide.properties['openslide.bounds-y'])
    except KeyError:
        offset_x = offset_y = 0
    
    level = int(np.log2(downsample))
    
    # If the pyramid is stored as 1, 2, 4, 8 etc.:
    if level == int(slide.get_best_level_for_downsample(downsample)):
        tile = slide.read_region((offset_x + coord[0], offset_y + coord[1]), level, (tile_size, tile_size))
        tile = np.asarray(tile)
    else:
        level = 0
        tile_size_fullres = tile_size * downsample

        tile = slide.read_region((offset_x + coord[0], offset_y + coord[1]), level, (tile_size_fullres, tile_size_fullres))
        tile = tile.resize((tile_size, tile_size))

    tile_array = np.array(tile)
    # Convert to grayscale for threshold calculations.
    gray_tile = cv2.cvtColor(tile_array, cv2.COLOR_BGRA2GRAY)

    # _, threshold_tile = cv2.threshold(gray_tile, 240, 255, cv2.THRESH_BINARY_INV)

    # tissue_pixels = np.count_nonzero(threshold_tile)
    # tissue_percentage = tissue_pixels / (threshold_tile.shape[0] * threshold_tile.shape[1]) * 100
    	
    tissue_pixels = np.count_nonzero(gray_tile < 225)
    tissue_percentage = (tissue_pixels / (gray_tile.shape[0] * gray_tile.shape[1])) * 100

    if tissue_percentage > 80:
        cv2.imwrite(os.path.join(output_path, f"{os.path.basename(output_path)}_{coord[0]}_{coord[1]}.jpg"),
                    cv2.cvtColor(tile_array, cv2.COLOR_RGBA2BGRA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    # Get input from commandline
    parsed_args = parse_arguments()

    base_dir = parsed_args.input

    image_annotation_paths = []

    images_dir = os.path.abspath(os.path.join(base_dir, 'Images'))
    annotations_dir = os.path.abspath(os.path.join(base_dir, 'Annotations'))

    extension = '.mrxs' if parsed_args.source == "EMC" else '.svs'

    for dirpath, _, filenames in os.walk(images_dir):
        for filename in filenames:
            if filename.endswith(extension):
                image_path = os.path.join(dirpath, filename)
                annotation_filename = filename.replace(extension, '.geojson')
                annotation_path = os.path.join(annotations_dir, os.path.relpath(dirpath, images_dir), annotation_filename)

                if os.path.exists(annotation_path):
                    image_annotation_paths.append((image_path, annotation_path))
    
    tile_size = (parsed_args.tilesize, parsed_args.tilesize)

    for slide_path, annotation_path in image_annotation_paths:
        output_path = os.path.join(parsed_args.output, os.path.basename(slide_path).replace(extension, ''))
        print(output_path)
        if os.path.isdir(output_path):
            print('Folder already exists, skipping')
            continue

        slide = openslide.open_slide(slide_path)

        objective_power = slide.properties['openslide.objective-power']

        downsample = int(int(objective_power) / parsed_args.magnification)
        if parsed_args.source == "EMC":
            downsample = 2 * downsample

        print(downsample)

        tiles_coords, num_tiles = tile_slide(annotation_path, tile_size, downsample)

        print(f'Processing slide {slide_path}: generating {num_tiles} tiles.')

        tar_path(output_path)

        # Multithreading is used to accelerate writing the extraction and saving of tiles.
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda coord: process_and_save_tile(slide, coord, tile_size[0], downsample, output_path),
                                   tiles_coords)))

        print('----' * 8)


if __name__ == "__main__":
    main()
