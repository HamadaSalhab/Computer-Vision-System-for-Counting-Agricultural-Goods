# DATASET LINK: "https://app.roboflow.com/ds/5wWd1OaqhW?key=io0HopkvZo"

import requests
import zipfile
import os
import pandas as pd

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
    print('fetching url...')
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

if __name__ == '__main__':
    get_dataset()
    