from utils.combined_normals import combinedNormals
import numpy as np
import torch

class SinusoidalData():
    def __init__(self, min_x=-10, max_x=10):
        self.min_x = min_x
        self.max_x = max_x

    def get_data_points(self, n_points):
        x_points = np.random.uniform(self.min_x, self.max_x, size=n_points)

        mu_values = self.mu_x(x_points)

        y_points = np.array([
            combinedNormals(-mu, 0.05, mu, 0.05).rvs()
            for mu in mu_values
        ])
        return (x_points, y_points)

    def mu_x(self, x):
        return 0.5 * np.sin(0.8 * x) + 0.05 * x
    
    def pdf(self, x):
        dist = combinedNormals(-self.mu_x(x), 0.05, self.mu_x(x), 0.05)
        dist.pdf(x)

    def pdf_list(self, values):
        pdf_values = np.array(self.pdf(values))
        return torch.from_numpy(pdf_values)
