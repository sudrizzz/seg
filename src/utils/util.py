import os
from os.path import abspath, exists

import matplotlib.pyplot as plt
import seaborn as sns


def makesure_dirs(paths) -> None:
    """
    检查路径里的文件夹是否存在，若不存在，则按层级依次新建文件夹
    :param paths: path string or path string list
    :return: None
    """
    if type(paths) == list:
        for path in paths:
            path = abspath(path)
            if not exists(path):
                os.makedirs(path, exist_ok=True)
    else:
        path = abspath(paths)
        if not exists(path):
            os.makedirs(path, exist_ok=True)


def plot_matrix(matrix, classes, x_label, y_label, save_to, ticks_rotation=45, show=False):
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(matrix, annot=True, cmap='crest')
    classes_indexes = classes.argsort()
    classes_labels = classes.tolist()
    ax.set_xticks(classes_indexes)
    ax.set_yticks(classes_indexes)
    ax.set_xticklabels(classes_labels, rotation=ticks_rotation, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(classes_labels, rotation=ticks_rotation, ha='right', rotation_mode='anchor')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_to, bbox_inches='tight', pad_inches=0.5)
    if show:
        plt.show()
