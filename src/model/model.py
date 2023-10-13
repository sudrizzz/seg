import datetime
import logging
from abc import ABC
from time import strftime

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from src.dataloader.custom_dataset import CustomDataset
from src.dataloader.dataloader import DataLoader
from src.model.base_model import BaseModel
from src.model.network import Network
from src.utils.util import makesure_dirs


class Model(BaseModel, ABC):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.train_sequence, self.val_sequence = None, None
        self.train_loss, self.val_loss = [], []

        self.save_to = '../' + self.config.model.save_to + strftime('%Y-%m-%d-%H-%M')
        makesure_dirs(self.save_to)

        logging.basicConfig(filename=self.save_to + '/train.log', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.logger.info(config)

    def load_data(self) -> None:
        data = DataLoader().load_data(self.config, inference=False)
        self.train_sequence, self.val_sequence = data[0], data[1]

    def build(self) -> None:
        self.model = Network(self.config)

    def train(self) -> None:
        best_loss = 100.
        self.model.to(self.device)
        self.logger.info(self.model)
        self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train.lr)

        train_dataloader = torch.utils.data.DataLoader(
            CustomDataset(self.train_sequence),
            batch_size=self.config.train.batch_size,
            shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            CustomDataset(self.val_sequence),
        )

        self.logger.info("Length of train_sequence: {}".format(len(self.train_sequence)))
        self.logger.info("Length of val_sequence: {}".format(len(self.val_sequence)))

        self.logger.info("===> Training started.")
        for epoch in range(self.config.train.epoch):
            train_loss_tmp = .0
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                x, y = batch['sequence'].to(self.device), batch['label'].to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                train_loss_tmp += loss.item()

                prediction = torch.argmax(output, dim=1)
                accuracy = (prediction == y).sum().item() / len(y)

                self.logger.info(
                    "===> Epoch [{:0>3d}/{:0>3d}] ({:0>3d}/{:0>3d}), Loss : {:.8f}, Accuracy : {:.4f}".format(
                        epoch + 1, self.config.train.epoch, step + 1, len(train_dataloader), loss.item(), accuracy
                    )
                )
            self.train_loss.append(train_loss_tmp / len(train_dataloader))

            # Validation
            val_loss, accuracy = .0, .0
            self.model.eval()
            with torch.no_grad():
                for _, batch in enumerate(val_dataloader):
                    x, y = batch['sequence'].to(self.device), batch['label'].to(self.device)
                    output = self.model(x)
                    loss = self.criterion(output, y)
                    val_loss += loss.item()

                    prediction = torch.argmax(output, dim=1)
                    accuracy += (prediction == y).sum().item()

                self.logger.info(
                    "===> Validation, Average Loss : {:.8f}, Accuracy : {:.4f}".format(
                        val_loss / len(val_dataloader), accuracy / len(val_dataloader)
                    )
                )

            self.val_loss.append(val_loss / len(val_dataloader))

            # Save best model
            if best_loss > self.val_loss[-1]:
                best_loss = self.val_loss[-1]
                self.save(epoch, self.model)

        self.logger.info("===> Training finished.")

    def save(self, epoch: int, model: torch.nn.Module) -> None:
        # revise plot index error
        epoch += 1
        model_path = self.save_to + '/best_model.pth'
        loss_path = self.save_to + '/loss.txt'
        fig_path = self.save_to + '/loss.png'

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'lr': self.model.optimizer.param_groups[0]['lr'],
        }, model_path)

        loss = np.column_stack((self.train_loss, self.val_loss))
        np.savetxt(loss_path, loss, fmt='%.8f', delimiter=',')

        plt.clf()
        plt.gcf().set_size_inches(8, 6)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(np.linspace(1, epoch, epoch).tolist(), self.train_loss, label='train loss')
        plt.plot(np.linspace(1, epoch, epoch).tolist(), self.val_loss, label='val loss')
        plt.legend(['train loss', 'val loss'])
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.clf()
        self.logger.info("===> Model saved to {}".format(model_path))
