from torch import nn, optim
import torch
import copy
import logging
from threading import Thread
from utils.data import get_data
from torch.utils.data import DataLoader
import threading

import numpy as np
from collections import OrderedDict
import math

import time

class Client:
    def __init__(self, config):
        self.config = config
        self.num = self.config.clients.total
        self.client_id = [i for i in range(0, self.num)]
        self.model = None
        self.dataloaders = []
        self.weights = []
        self.epoch_loss = []
        self.running_corrects = []
        self.len_dataset = []
        self.training_time = []

    def load_data(self):
        self.trainset, self.testset = get_data(self.config.dataset, self.config)
        for subset in self.trainset:
            loader = DataLoader(subset, batch_size=self.config.fl.batch_size)
            self.dataloaders.append(loader)

    def clients_to_server(self):
        return self.client_id

    def get_model(self, model):
        self.model = model

    def quantize(self, v, s):
        norm = np.sqrt(np.sum(np.square(v)))
        level_float = s * np.abs(v) / norm
        previous_level = np.floor(level_float)
        is_next_level = np.random.rand(*v.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level
        
        return np.sign(v) * norm * new_level / s

    def local_train(self, user_id, dataloaders, verbose=1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model)
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.config.fl.lr)

        for e in range(self.config.fl.epochs):
            start = time.time()
            running_loss = 0
            running_corrects = 0
            epoch_loss = 0
            epoch_acc = 0
            training_time = 0.0

            for inputs, labels in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = int(running_corrects) / len(dataloaders.dataset)
            end = time.time()
            training_time = (end-start)
            logging.debug('User {}: {} Loss: {:.4f} Acc: {:.4f}'.format(user_id, "training", epoch_loss, epoch_acc))

        # need be locked
        lock = threading.Lock()
        lock.acquire()

        # quantization 16 bits
        d = OrderedDict()
        for k in model.state_dict().keys():
            q = self.quantize(model.state_dict()[k].numpy(), 2**16)
            d[k] = torch.Tensor(q)

        self.weights.append(d)  #copy.deepcopy(model.state_dict()))
        self.epoch_loss.append(epoch_loss)
        self.training_time.append(math.log10(training_time))
        self.running_corrects.append(int(running_corrects))
        self.len_dataset.append(len(dataloaders.dataset))
        lock.release()

    def upload(self, info):
        return info

    def update(self, glob_weights):
        self.model.load_state_dict(glob_weights)

    def train(self, selected_client):
        self.weights = []
        self.epoch_loss = []
        self.running_corrects = []
        self.training_time = []
        self.len_dataset = []

        # multithreading
        threads = [Thread(target=self.local_train(user_id=client, dataloaders=self.dataloaders[client])) for client in
                   selected_client]
        [t.start() for t in threads]
        [t.join() for t in threads]
        # training details
        info = {"weights": self.weights, "loss": self.epoch_loss, "corrects": self.running_corrects,
                "len": self.len_dataset, "training_time": self.training_time}
        return self.upload(info)

    def test(self):
        corrects = 0
        test_loss = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = copy.deepcopy(self.model)
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(self.testset, batch_size=32, shuffle=True)
        for batch_id, (inputs, labels) in enumerate(dataloader):
            loss = 0
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        acc = int(corrects) / len(dataloader.dataset)
        avg_loss = test_loss / len(dataloader.dataset)

        return acc, avg_loss


if __name__ == "__main__":
    c = Client(100)
