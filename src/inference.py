import argparse
import datetime
import logging

import numpy as np
import torch

from config.config import CFG
from dataloader.custom_dataset import CustomDataset
from dataloader.dataloader import DataLoader
from model.network import Network
from utils.config import Config


def inference(folder) -> None:
    """
    Evaluate model
    """
    config = Config.from_json(CFG)
    folder = '../' + config.model.save_to + folder
    logging.basicConfig(filename=folder + '/test.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('===> Inference started.')
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

    inference_sequence = DataLoader().load_data(config, inference=True)[0]
    logger.info('Length of inference_sequence: {}'.format(len(inference_sequence)))
    inference_dataloader = torch.utils.data.DataLoader(
        CustomDataset(inference_sequence),
    )

    result = []
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(inference_dataloader):
            x, y = batch['sequence'].to(device), batch['label'].to(device)
            output = model(x)
            prediction = torch.argmax(output, dim=1)
            result.append(str(prediction.item()))

    with open(folder + '/inference.txt', 'w') as f:
        f.writelines('\n'.join(result))
    logger.info('===> Inference finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-folder', default=None, type=str, required=True,
                        help='name of folder which is under the folder \'saved\' '
                             'and contains \'best_model.pth\' (eg: \'2023-01-01-00-00-00\')')
    args = parser.parse_args()
    inference(folder=args.model_folder)
