import os

import numpy as np
import SimpleITK as sitk
from PIL import Image
from pathlib import Path

from config.config import CFG
from utils.config import Config
from utils.util import makesure_dirs


def process_raw_data(config):
    config = Config.from_json(config)
    raw_data_dir = Path(config.data.raw)
    patient_list = os.listdir(raw_data_dir)
    img_path_list, gt_path_list = [], []

    for i, patient_name in enumerate(patient_list):
        # Exclude hidden folders
        if patient_name.startswith('.'):
            continue
        save_img(config, raw_data_dir, patient_name, view='2CH', instant='ES',
                 img_path_list=img_path_list, gt_path_list=gt_path_list)
        save_img(config, raw_data_dir, patient_name, view='2CH', instant='ED',
                 img_path_list=img_path_list, gt_path_list=gt_path_list)
        save_img(config, raw_data_dir, patient_name, view='4CH', instant='ES',
                 img_path_list=img_path_list, gt_path_list=gt_path_list)
        save_img(config, raw_data_dir, patient_name, view='4CH', instant='ED',
                 img_path_list=img_path_list, gt_path_list=gt_path_list)
        print(f'Processed {i}/{len(patient_list)}')

    processed_data_dir = Path(config.data.processed)
    with open(processed_data_dir / 'img_path_list.txt', 'w') as f:
        for item in img_path_list:
            f.write("%s\n" % item)
    with open(processed_data_dir / 'gt_path_list.txt', 'w') as f:
        for item in gt_path_list:
            f.write("%s\n" % item)


def save_img(config, raw_data_dir: Path, patient_name: str, view: str, instant: str,
             img_path_list: list, gt_path_list: list):
    patient_dir = raw_data_dir / patient_name
    img_pattern = '{patient_name}_{view}_{instant}.nii.gz'
    gt_pattern = '{patient_name}_{view}_{instant}_gt.nii.gz'

    # Load image and save info
    input_image = sitk.ReadImage(
        str(patient_dir / img_pattern.format(patient_name=patient_name, view=view, instant=instant)))
    gt_image = sitk.ReadImage(
        str(patient_dir / gt_pattern.format(patient_name=patient_name, view=view, instant=instant)))

    # Extract numpy array from the SimpleITK image object
    img_array = sitk.GetArrayFromImage(input_image)
    gt_array = sitk.GetArrayFromImage(gt_image)

    # Expand gray scale range to 0-255
    img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    gt_array = np.where(gt_array == 2, 255, 0).astype(np.uint8)

    processed_data_dir = Path(config.data.processed) / patient_name
    makesure_dirs(processed_data_dir)

    img_path = (
        str(processed_data_dir / img_pattern.format(patient_name=patient_name, view=view, instant=instant))
        .replace('.nii.gz', '.png')
    )
    gt_path = (
        str(processed_data_dir / gt_pattern.format(patient_name=patient_name, view=view, instant=instant))
        .replace('.nii.gz', '.png')
    )

    Image.fromarray(img_array).save(img_path)
    Image.fromarray(gt_array).save(gt_path)
    img_path_list.append(img_path)
    gt_path_list.append(gt_path)


if __name__ == '__main__':
    process_raw_data(CFG)
