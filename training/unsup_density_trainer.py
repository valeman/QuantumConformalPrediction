# model_a_trainer.py
from base_trainer import BaseTrainer
from metrics import NegativeLogSumCriterion
from torch.utils.data import DataLoader
import torch.optim as optim
from models.utils import calculate_probabilities

class UnsupDensityModelTrainer(BaseTrainer):
    def __init__(self, pqc, q_device, dataset, batch_size, optimizer=None, criterion=None):
        if optimizer is None: optimizer = optim.Adam(pqc.parameters(), lr=0.01)
        if criterion is None: criterion = NegativeLogSumCriterion()
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        super().__init__(pqc, q_device, optimizer, criterion, data_loader)

    def train_one_epoch(self, batch_samples):
        for batch_samples in self.data_loader:
            output = self.pqc(q_device=self.q_device)
            model_probabilities = calculate_probabilities(output, batch_samples[0])
            loss = self.criterion(model_probabilities)
            loss.backward()
            self.optimizer.step()

    def train(self, n_epochs=100, plot_loss=False):
        self.optimizer.zero_grad()
        for epoch in range(n_epochs):
            self.train_one_epoch()
        return self.pqc