# DATASET LINK: "https://app.roboflow.com/ds/5wWd1OaqhW?key=io0HopkvZo"

import requests
import zipfile
import os
import pandas as pd
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import torchvision



# Define the dataset URL and the paths for storing the dataset
DATASET_DOWNLOAD_URL = "https://app.roboflow.com/ds/5wWd1OaqhW?key=io0HopkvZo"
DATASET_RAW_PATH_RELATIVE = '/../../data/raw/blueberries.zip'
DATASET_RAW_PATH_ABSOLUTE = os.path.dirname(__file__) + DATASET_RAW_PATH_RELATIVE
DATASET_RAW_DIR_RELATIVE = '/../../data/raw/blueberries/'
DATASET_RAW_DIR_ABSOLUTE = os.path.dirname(__file__) + DATASET_RAW_DIR_RELATIVE

def download_dataset(url, save_path, chunk_size=128):
    """
    Downloads a dataset from a specified URL and saves it to a path.

    This function streams the dataset in chunks to avoid high memory usage when downloading large files.

    Args:
        url (str): The URL from which to download the dataset.
        save_path (str): The path to which the dataset should be saved.
        chunk_size (int): The size of the chunks to stream while downloading, in bytes.

    Raises:
        requests.exceptions.RequestException: If there is an issue with network access or the request.
    """
    print('Fetching url...')
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        print('Fetched Successfully.')
        with open(save_path, 'wb') as fd:
            print('Writing... (this might take a while)')
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
        print("Finished downloading successfully.")
    else:
        r.raise_for_status()

def extract_zip(zip_path, destination_dir):
    """
    Extracts a zip file to a specified destination directory.

    Args:
        zip_path (str): The path of the zip file to extract.
        destination_dir (str): The directory to which the contents of the zip should be extracted.

    Raises:
        zipfile.BadZipFile: If the file is not a zip file or it is corrupted.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print('Extracting...')
        zip_ref.extractall(destination_dir)
    print('Finished extracting.')

def get_dataset():
    """
    Ensures the dataset is downloaded and extracted in the designated directory.

    This function checks if the dataset already exists, if not, it downloads and extracts it.
    
    Raises:
        Exception: If there is an issue accessing the file system or handling the zip file.
    """
    try:
        # Directory exists, and not empty
        if os.path.isdir(DATASET_RAW_DIR_ABSOLUTE) and len(os.listdir(DATASET_RAW_DIR_ABSOLUTE)) != 0:
            print('Dataset already exists at ' + DATASET_RAW_DIR_ABSOLUTE)
        elif os.path.exists(DATASET_RAW_PATH_ABSOLUTE):
            print(f"Dataset doesn't exist at {DATASET_RAW_DIR_ABSOLUTE}, but .zip file is found")
            extract_zip(zip_path=DATASET_RAW_PATH_ABSOLUTE, destination_dir=DATASET_RAW_DIR_ABSOLUTE)
        else:
            print('Dataset does not exist, downloading...')
            download_dataset(DATASET_DOWNLOAD_URL, DATASET_RAW_PATH_ABSOLUTE)
            extract_zip(zip_path=DATASET_RAW_PATH_ABSOLUTE, destination_dir=DATASET_RAW_DIR_ABSOLUTE)
        
    except Exception as e:
        print('An error occurred: ' + str(e))
        raise e
    finally:
        print('All set. The dataset can be found in project_root_dir/data/raw.')


def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou

def resize_image(image):
    # Get the current dimensions of the image
    height, width = image.shape[:2]

    # Determine the maximum dimension
    max_dim = max(height, width)

    # Calculate the scale factor to resize the image
    scale = 224 / max_dim

    # Calculate the new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the image while preserving the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def crop_image(xmin, ymin, xmax, ymax, image):
    # Calculate the original width and height of the bounding box
    box_width = xmax - xmin
    box_height = ymax - ymin

    # Calculate the center coordinates of the bounding box
    center_x = int((xmin + xmax) / 2)
    center_y = int((ymin + ymax) / 2)

    # Calculate the size of the largest dimension
    max_dim = max(box_width, box_height)

    # Calculate the new bounding box coordinates
    new_xmin = max(center_x - max_dim // 2, 0)
    new_ymin = max(center_y - max_dim // 2, 0)
    new_xmax = new_xmin + max_dim
    new_ymax = new_ymin + max_dim

    # Crop the new bounding box region from the image
    cropped_image = resize_image(image[new_ymin:new_ymax, new_xmin:new_xmax])
    new_image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Calculate the position to place the object in the center
    x_offset = int((224 - cropped_image.shape[1]) / 2)
    y_offset = int((224 - cropped_image.shape[0]) / 2)

    # Paste the cropped image in the center of the new image
    new_image[y_offset:y_offset+cropped_image.shape[0], x_offset:x_offset+cropped_image.shape[1]] = cropped_image
    return new_image

def prepare_dataset():

    annotations_df = pd.read_csv(f'../../data/raw/blueberries/train/_annotations.csv')
    dataset_path = f'../../data/raw/blueberries/train/'
    destination_dir = '../../data/interim/train/'
    sam_checkpoint = "../../data/external/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    if torch.cuda.is_available():
        sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)


    groups = annotations_df.groupby("filename")
    print("Started processing images...")
    for image_name, group in groups:
        image = cv2.imread(dataset_path + "/" + image_name)
        masks = mask_generator.generate(image)
        for index, mask in enumerate(masks):
            masked_image = image * mask["segmentation"][:, :, np.newaxis]
            cropped_masked_image = crop_image(mask["bbox"][0], mask["bbox"][1],
                                              mask["bbox"][0] + mask["bbox"][2],
                                              mask["bbox"][1] + mask["bbox"][3], masked_image)

            ground_truth_boxes = group[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
            iou = 0
            for box in ground_truth_boxes:
                box1 = box
                box2 = [mask['bbox'][0], mask['bbox'][1], mask['bbox'][0] + mask['bbox'][2],
                        mask['bbox'][1] + mask['bbox'][3]]
                iou = max(get_iou(box1, box2), iou)
            if iou > 0.65:
                save_path = f'{destination_dir}/cropped_images_positive/{image_name}_object_{index}.jpg'
                cv2.imwrite(save_path, cropped_masked_image)

            else:
                save_path = f'{destination_dir}/cropped_images_negative/{image_name}_object_{index}.jpg'
                cv2.imwrite(save_path, cropped_masked_image)
    print("Finished processing. The processed masked images can be found in project_root_dir/data/interim/train.")
if __name__ == '__main__':
    get_dataset()
    prepare_dataset()
    # pd.read_csv('../../data/raw/blueberries/train/_annotations.csv')
    