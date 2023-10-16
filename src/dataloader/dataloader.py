from pathlib import Path

import cv2
import numpy as np


class DataLoader(object):
    def __init__(self, config):
        self.x_train, self.y_train, self.x_test = None, None, None
        self.config = config

    def load_data(self, inference: bool = False):
        images = np.loadtxt(Path(self.config.data.processed) / 'img_path_list.txt', dtype=str)
        masks = np.loadtxt(Path(self.config.data.processed) / 'gt_path_list.txt', dtype=str)
        return self.train_val_test_split(images=images,
                                         masks=masks,
                                         train_portion=self.config.train.train_portion,
                                         val_portion=self.config.train.val_portion,
                                         shuffle=self.config.train.shuffle,
                                         inference=inference)

    def train_val_test_split(self,
                             images: np.ndarray,
                             masks: np.ndarray,
                             train_portion: float = 0.7,
                             val_portion: float = 0.2,
                             shuffle: bool = False,
                             inference: bool = False) -> dict:
        """
        Split dataset into train, val and test dataset
        """
        if shuffle:
            np.random.seed(self.config.train.seed)
            permutation = np.random.permutation(len(images))
            np.savetxt(Path(self.config.data.processed) / 'permutation.txt', permutation)
            images = images[permutation]
            masks = masks[permutation]

        train_size = int(len(images) * train_portion)
        val_size = int(len(images) * val_portion)

        if not inference:
            return {
                'train': {
                    'images': images[:train_size],
                    'masks': masks[:train_size]
                },
                'val': {
                    'images': images[train_size:train_size + val_size],
                    'masks': masks[train_size:train_size + val_size]
                }
            }
        else:
            return {
                'test': {
                    'images': images[train_size + val_size:],
                    'masks': masks[train_size + val_size:]
                }
            }
