from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from training.base_trainer import BaseTrainer
from training.metrics import NegativeLogSumCriterion
from circuits.utils import calculate_probabilities

class BackpropogationTrainer(BaseTrainer):
    def __init__(self, pqc, q_device, dataset, batch_size, optimizer=None, criterion=None):
        if optimizer is None: optimizer = optim.Adam(pqc.parameters(), lr=0.01)
        if criterion is None: criterion = NegativeLogSumCriterion()
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        super().__init__(pqc, q_device, optimizer, criterion, data_loader)

    def train_one_epoch(self):
        total_loss = 0
        for batch_samples in self.data_loader:
            output = self.pqc(q_device=self.q_device, measure=True)
            model_probabilities = calculate_probabilities(output, batch_samples[0])
            loss = self.criterion(model_probabilities)
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