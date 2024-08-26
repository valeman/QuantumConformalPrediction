from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from training.base_trainer import BaseTrainer
from training.metrics import NegativeLogSumCriterion
from circuits.utils import calculate_expectation
import torch
import torch.nn as nn
from utils.helper_functions import evenlySpaceEigenstates
from circuits.hardware_eff_no_input import *

class BackpropogationTrainer(BaseTrainer):
    def __init__(self, pqc, q_device, dataset, batch_size, eigenvalues, optimizer=None, criterion=None):
        if optimizer is None: optimizer = optim.Adam(pqc.parameters(), lr=0.01)
        if criterion is None: criterion = nn.MSELoss()
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        super().__init__(pqc, q_device, optimizer, criterion, data_loader)
        self.eigenvalues = eigenvalues

    def train_one_epoch(self):
        total_loss = 0
        for batch_samples in self.data_loader:
            expectation = self.pqc.calculate_expected_value(self.eigenvalues)
            loss = self.criterion(expectation.repeat(batch_samples[0].size(0)), evenlySpaceEigenstates(batch_samples[0], self.pqc.n_wires, -1.5, 1.5))
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return total_loss
        

    def train(self, n_epochs=100, plot_loss=False):
        losses = []
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            epoch_loss = self.train_one_epoch()
            losses.append(epoch_loss)
            print(f"Training Epoch: {epoch}/{n_epochs}", end='\r')

        # Plotting the loss over epochs
        if plot_loss:
            plt.plot(range(n_epochs), losses)
            plt.xlabel('Epoch')
            plt.ylabel('Batch Agv Loss')
            plt.title('Loss over Epochs')
            plt.show()

        return self.pqc