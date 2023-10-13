import argparse
import datetime
import logging

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from torch import nn

from config.config import CFG
from dataloader.custom_dataset import CustomDataset
from dataloader.dataloader import DataLoader, label_classes
from model.network import Network
from utils.config import Config
from utils.util import plot_matrix


def evaluate(folder) -> None:
    """
    Evaluate model
    """
    config = Config.from_json(CFG)
    folder = '../' + config.model.save_to + folder
    logging.basicConfig(filename=folder + '/test.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('===> Testing started.')
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(CFG)

    # fix random seeds for reproducibility
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(config.train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = Network(config)
    pth = torch.load(folder + '/best_model.pth')
    model.load_state_dict(pth['model'])
    model.to(device)
    logger.info(model)

    test_sequence = DataLoader().load_data(config, inference=False)[2]
    logger.info('Length of test_sequence: {}'.format(len(test_sequence)))
    test_dataloader = torch.utils.data.DataLoader(
        CustomDataset(test_sequence),
    )

    model.eval()
    result, y_true, y_pred = [], [], []
    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            x, y = batch['sequence'].to(device), batch['label'].to(device)
            y_true.append(y.item())
            output = model(x)
            prediction = torch.argmax(output, dim=1)
            y_pred.append(prediction.item())
            result.append(str(prediction.item()))

    with open(folder + '/test_result.txt', 'w') as f:
        f.writelines('\n'.join(result))

    classes = label_classes()
    matrix = confusion_matrix(y_true, y_pred, labels=classes.argsort())

    np.savetxt(folder + '/confusion_matrix.txt', matrix, fmt='%d', delimiter=',')
    plot_matrix(matrix, classes, x_label='Predicted Label', y_label='True Label',
                save_to=folder + '/confusion_matrix.pdf', ticks_rotation=30, show=True)

    average = 'micro'
    test_metrics = ['accuracy:\t' + str(accuracy_score(y_true, y_pred)),
                    'precision:\t' + str(precision_score(y_true, y_pred, average=average)),
                    'recall: \t' + str(recall_score(y_true, y_pred, average=average)),
                    'f1_score:\t' + str(f1_score(y_true, y_pred, average=average))]
    with open(folder + '/test_metrics.txt', 'w') as f:
        f.writelines('\n'.join(test_metrics))

    logger.info('===> Testing finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-folder', default=None, type=str, required=True,
                        help='name of folder which is under the folder \'saved\' '
                             'and contains \'best_model.pth\' (eg: \'2023-01-01-00-00-00\')')
    args = parser.parse_args()
    evaluate(folder=args.model_folder)
