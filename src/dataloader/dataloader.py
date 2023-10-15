from pathlib import Path

import numpy as np


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


class DataLoader(object):
    def __init__(self, config):
        self.x_train, self.y_train, self.x_test = None, None, None
        self.config = config

    def load_data(self, inference: bool = False):
        img_path_list = np.loadtxt(Path(self.config.data.processed) / 'img_path_list.txt', dtype=str)
        gt_path_list = np.loadtxt(Path(self.config.data.processed) / 'gt_path_list.txt', dtype=str)
        # draw_contours(img_path_list[0], gt_path_list[0])
        pass

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
