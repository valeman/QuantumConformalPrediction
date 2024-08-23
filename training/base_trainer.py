class BaseTrainer:
    def __init__(self, pqc, q_device, optimizer, criterion, data_loader):
        self.pqc = pqc
        self.q_device = q_device
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_loader = data_loader

    def train_one_epoch(self):
        pass

    def train(self):
        pass