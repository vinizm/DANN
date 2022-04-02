import numpy as np
from abc import ABC, abstractmethod

from utils.hyperparameters import LR_WARMUP, LR0, ALPHA, BETA, STEP_DECAY, NUM_STEPS


class AbstractLearningRate(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def calculate(self):
        pass


class LearningRateConstant(AbstractLearningRate):

    def __init__(self, const = LR0):
        self.const = const

    def calculate(self, p: float):
        return self.const

class LearningRateExpDecay(AbstractLearningRate):

    def __init__(self, lr0 = LR0, lr_warmup = LR_WARMUP, alpha = ALPHA, beta = BETA):
        self.lr0 = lr0
        self.lr_warmup = lr_warmup
        self.alpha = alpha
        self.beta = beta

    def calculate(self, p: float):
        if p <= self.lr_warmup:
            return self.lr0
        return self.lr0 / ((1 + self.alpha * (p - self.lr_warmup)) ** self.beta)

class LearningRateStepDecay(AbstractLearningRate):

    def __init__(self, lr0 = LR0, decay = STEP_DECAY, num_steps = NUM_STEPS):
        self.lr0 = lr0
        self.decay = decay
        self.num_steps = num_steps

        self.points = np.linspace(0, 1, self.num_steps + 1)

    def calculate(self, p: float):
        
        for i in range(len(self.points) - 1):
            if p < self.points[i + 1]:

                return self.lr0 / (self.decay ** (i - 1))

# linear decay

# exponential growth

# linear growth

# step growth

class LearningRateFactory:

    @staticmethod
    def get_function(name: str, **kargs):

        if name == 'constant':
            return LearningRateConstant(**kargs)

        elif name == 'exp_decay':
            return LearningRateExpDecay(**kargs)

        elif name == 'step_decay':
            return LearningRateStepDecay(**kargs)
