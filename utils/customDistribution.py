import numpy as np
from scipy.stats import norm, rv_continuous
import torch

class combinedNormals(rv_continuous):
    def __init__(self, mean1, std1, mean2, std2):
        super().__init__()
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2
    
    def _pdf(self, x):
        pdf1 = norm.pdf(x, self.mean1, self.std1)
        pdf2 = norm.pdf(x, self.mean2, self.std2)
        return 0.5 * (pdf1 + pdf2)

    
    def _cdf(self, x):
        cdf1 = norm.cdf(x, self.mean1, self.std1)
        cdf2 = norm.cdf(x, self.mean2, self.std2)
        return 0.5 * (cdf1 + cdf2)
    
    def _rvs(self, size=None, random_state=None):
        choices = np.random.choice([0, 1], size=size, p=[0.5, 0.5])

        samples = np.where(
            choices == 0, 
            np.random.normal(self.mean1, self.std1, size),
            np.random.normal(self.mean2, self.std2, size)
        )
        return samples

    def pdf_list(self, values):
        pdf_values = np.array(self.pdf(values))
        return torch.from_numpy(pdf_values)




