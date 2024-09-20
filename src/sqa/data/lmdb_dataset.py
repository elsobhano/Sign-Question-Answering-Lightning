import lmdb
import cv2
import os
import numpy as np
import pickle
from tqdm import tqdm
import shutil
from pathlib import Path
from time import time
import torch

def create_lmdb_for_each_folder(main_folder, output_folder, map_size=1e12):
    """
    Create separate LMDB files for each subfolder in the main folder.

    Parameters:
    - main_folder (str): Path to the main folder containing subfolders with images.
    - output_folder (str): Path to the output folder where the LMDB databases will be stored.
    - map_size (int, optional): Maximum size of the LMDB database for each subfolder.
    """
    Path(output_folder).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # if Path(output_folder).exists():
    #     shutil.rmtree(output_folder)

    # For condor usage, we create a local database on the disk.
    # tmp_dir = os.path.join("/tmp", phase)
    # Path(tmp_dir).mkdir(parents=True)

    # Ensure the output directory exists

    # Iterate over each subfolder in the main folder
    folder_names = sorted([f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))])
    # return 0


    for folder_name in tqdm(folder_names):
        folder_path = os.path.join(main_folder, folder_name)
        lmdb_path = os.path.join(output_folder, f"{folder_name}.lmdb")  # Create a unique LMDB for each folder

        # Open LMDB environment for the current folder
        env = lmdb.open(lmdb_path, map_size=int(map_size))

        # Begin an LMDB write transaction
        txn = env.begin(write=True)

        # List all images in the current folder
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        # print(image_files)
        # exit()
        images_in_folder = []

        for idx, image_name in enumerate(image_files):
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path)  # Load image using OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = np.transpose(img, (2, 0, 1))  # Convert from HWC to CHW format
            images_in_folder.append(img)

        # Serialize the image
        img_data = pickle.dumps(images_in_folder)
        # Store in the LMDB database with image name as key
        txn.put("data".encode('ascii'), img_data)

        # Commit and close the LMDB transaction for the current folder
        txn.commit()
        env.close()
        # break

    # Move the database to its destination.
    # shutil.move(tmp_dir, output_folder)
    # Remove the temporary directories.
    # shutil.rmtree(tmp_dir)
    print(f"LMDB databases created for all subfolders in {output_folder}.")

def read_lmdb_folder(lmdb_path, folder_name=None):
    """
    Read images from a specific folder key in the LMDB database.

    Parameters:
    - lmdb_path (str): Path to the LMDB database.
    - folder_name (str): The key (folder name) to retrieve images from.
    """
    # print(list_all_keys(lmdb_path))
    if folder_name == None:
        lmdb_file = lmdb_path
    else:
        lmdb_file = os.path.join(lmdb_path, f"{folder_name}.lmdb")
    
    env = lmdb.open(lmdb_file, readonly=True)
    with env.begin() as txn:
        images_data = txn.get("data".encode('ascii'))

    # Deserialize the list of images
    images = pickle.loads(images_data)

    # Convert back from CHW to HWC format for visualization
    # images = [np.transpose(img, (1, 2, 0)) for img in images]

    return images

if __name__ == "__main__":
    
    data_path = "src/sqa/data"
    for phase in ["train", "dev", "test"]:
        phase_folder = f"src/sqa/data/fullFrame-210x260px/{phase}"
        lmdb_path = os.path.join(data_path,'lmdb',phase)
        # print(lmdb_path)
        create_lmdb_for_each_folder(phase_folder, lmdb_path)

    # images = read_lmdb_folder("src/sqa/data/lmdb/test/","01April_2010_Thursday_heute-6704")
    # torch_images = torch.from_numpy(np.stack(images))

    # print(type(images))
    # print(len(images))
    # print(type(images[0]))
    # print(torch_images.shape)

