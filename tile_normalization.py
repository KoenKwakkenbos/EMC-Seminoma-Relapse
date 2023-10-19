import os
import argparse
import numpy as np
import cv2
from helpers import dir_path, tar_path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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

    return parser.parse_args()


# Adapted from: https://github.com/schaugf/HEnorm_python
def _normalize_image(img, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[0.5626, 0.2159],
                    [0.7201, 0.8012],
                    [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, _ = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(float)+1)/Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    #project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1],That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    return Inorm

def process_image(input_path, output_path):
    # Function to process each image
    tile = cv2.imread(input_path)
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    tile = _normalize_image(tile)
    cv2.imwrite(output_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))


def main():
    parsed_args = parse_arguments()

    input_dir = parsed_args.input
    output_dir = parsed_args.output
    image_paths = []

    # Iterate through folders in the input path
    for dirpath, _, filenames in os.walk(input_dir):
        if dirpath == input_dir:
            continue

        # Create a similarly named folder in the output path
        relative_path = os.path.relpath(dirpath, input_dir)
        new_folder = os.path.join(output_dir, relative_path)
        # new_folder_normalized = os.path.join(new_folder)

        # Create the directory structure if it doesn't exist
		
        if os.path.isdir(new_folder):
            continue
		
        os.makedirs(new_folder, exist_ok=True)
        img_paths = []
        output_paths = []

        # Iterate through all images in the folder and normalize them
        for _, _, filenames in os.walk(dirpath):
            print(f"Processing {dirpath}, {len(filenames)} images")
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_path = os.path.join(dirpath, filename)
                    image_name = os.path.basename(image_path)
                    output_path = os.path.join(new_folder, image_name.replace('png', 'jpg'))

                    img_paths.append(image_path)
                    output_paths.append(output_path)
                    # process_image(image_path, output_path)

        # Use ThreadPoolExecutor to process images in parallel
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda img, out: process_image(img, out), img_paths, output_paths)))


    # with ThreadPoolExecutor() as executor:
    #     for root, _, files in os.walk(input_dir):
    #         for file in files:
    #             input_path = os.path.join(root, file)
    #             relative_path = os.path.relpath(input_path, input_dir)
    #             output_path = os.path.join(output_dir, relative_path)
    #             # Ensure the output directory exists
    #             os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #             # tile = cv2.imread(input_path)
    #             # tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

    #             # tile = _normalize_image(tile)

    #             # cv2.imwrite(output_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
    #             executor.submit(process_image, input_path, output_path)


if __name__ == "__main__":
    main()
