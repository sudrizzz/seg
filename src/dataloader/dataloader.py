import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import SimpleITK as sitk


def train_val_test_split(data: list,
                         train_portion: float = 0.6,
                         val_portion: float = 0.2,
                         shuffle: bool = False, seed: int = 42) -> (list, list, list):
    """
    Split dataset into train, val and test dataset
    """
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(data)
    train_size = int(len(data) * train_portion)
    val_size = int(len(data) * val_portion)
    return data[:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]


def read_img(raw_data_dir: Path, patient_name: str, view: str, instant: str) -> np.ndarray:
    patient_dir = raw_data_dir / patient_name
    gt_pattern = '{patient_name}_{view}_{instant}_gt.nii.gz'
    img_pattern = '{patient_name}_{view}_{instant}.nii.gz'

    # Load image and save info
    gt_image = sitk.ReadImage(
        str(patient_dir / gt_pattern.format(patient_name=patient_name, view=view, instant=instant)))
    input_image = sitk.ReadImage(
        str(patient_dir / img_pattern.format(patient_name=patient_name, view=view, instant=instant)))

    # Extract numpy array from the SimpleITK image object
    gt_array = sitk.GetArrayFromImage(gt_image)
    img_array = sitk.GetArrayFromImage(input_image)

    # Expand gray scale range to 0-255
    gt_array = ((gt_array - gt_array.min()) / (gt_array.max() - gt_array.min()) * 255).astype(np.uint8)
    img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_array, cmap='gray')
    ax[1].imshow(gt_array, cmap='gray')
    plt.show()

    # Image.fromarray(gt_array).show()
    # Image.fromarray(img_array).show()

    return np.stack([gt_array, img_array], axis=0)


class DataLoader(object):
    def __init__(self):
        self.x_train, self.y_train, self.x_test = None, None, None

    def load_data(self, config: object, inference: bool = False) -> list:
        raw_data_dir = Path(config.data.raw)
        patient_list = os.listdir(raw_data_dir)

        for patient_name in patient_list:
            # exclude hidden folders
            if patient_name.startswith('.'):
                continue

            img_2ch_es = read_img(raw_data_dir, patient_name, view='2CH', instant='ES')
            img_2ch_ed = read_img(raw_data_dir, patient_name, view='2CH', instant='ED')
            img_4ch_es = read_img(raw_data_dir, patient_name, view='4CH', instant='ES')
            img_4ch_ed = read_img(raw_data_dir, patient_name, view='4CH', instant='ED')

            print()

        # return self.process(config, inference)

    def process(self, config: object, inference: bool = False) -> list:
        # self.y_train.surface.value_counts().plot(kind='bar')
        # plt.xticks(rotation=45)
        # plt.show()

        if inference:
            label = -1
            inference_sequence = []
            feature_columns = self.x_test.columns.tolist()[3:]
            for series_id, group in self.x_test.groupby('series_id'):
                sequence_feature = group[feature_columns]
                inference_sequence.append((sequence_feature, label))
            return [inference_sequence]

        else:
            pass

            # return [train_sequence, val_sequence, test_sequence]
