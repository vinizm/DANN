import numpy as np
from abc import ABC, abstractmethod

from utils.hyperparameters import *


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

    def __init__(self, lr0 = LR0, step_decay = STEP_DECAY, num_steps = NUM_STEPS, warmup = LR_WARMUP):
        self.lr0 = lr0
        self.step_decay = step_decay
        self.num_steps = num_steps
        self.warmup = warmup

        self.points = np.linspace(0, 1, self.num_steps + 1)

    def calculate(self, p: float):
        
        for i in range(len(self.points) - 1):
            if p <= self.points[i + 1] + 1e-6:

                return self.lr0 / (self.step_decay ** i)

class LearningRateLinear(AbstractLearningRate):

    def __init__(self, start: float = LR_START_LINEAR, stop: float = LR_STOP_LINEAR):

        self.start = start
        self.stop = stop

        self.a = stop - start
        self.b = start

    def calculate(self, p: float):
        return self.a * p + self.b

class LearningRateLog(AbstractLearningRate):

    def __init__(self, start: float = LR_START_LOG, stop: float = LR_STOP_LOG):

        self.start = start
        self.stop = stop

        self.a = stop - start
        self.b = start

    def calculate(self, p: float):
        return 10 ** (self.a * p + self.b)


class LearningRateFactory:

    @staticmethod
    def get_function(name: str, **kwargs):

        if name == 'constant':
            return LearningRateConstant(**kwargs)

        elif name == 'exp_decay':
            return LearningRateExpDecay(**kwargs)

        elif name == 'step':
            return LearningRateStepDecay(**kwargs)

        elif name == 'linear':
            return LearningRateLinear(**kwargs)

        elif name == 'log':
            return LearningRateLog(**kwargs)
