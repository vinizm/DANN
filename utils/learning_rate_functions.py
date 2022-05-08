import numpy as np
from abc import ABC, abstractmethod

from utils.hyperparameters import *


class AbstractLearningRate(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def calculate(self):
        pass

    @property
    @abstractmethod
    def config(self):
        pass


class LearningRateConstant(AbstractLearningRate):

    def __init__(self, const = LR0):
        self.const = const

    def calculate(self, p: float):
        return self.const

    @property
    def config(self):
        config = {'const': self.const}
        return config

class LearningRateExpDecay(AbstractLearningRate):

    def __init__(self, lr0 = LR0, warmup = LR_WARMUP, alpha = ALPHA, beta = BETA):
        self.lr0 = lr0
        self.warmup = warmup
        self.alpha = alpha
        self.beta = beta
        self.name = 'exp_decay'

    def calculate(self, p: float):
        if p <= self.warmup:
            return self.lr0
        return self.lr0 / ((1 + self.alpha * (p - self.warmup)) ** self.beta)

    @property
    def config(self):
        config = {
                    'lr0': self.lr0,
                    'warmup': self.warmup,
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'name': self.name
                }
        return config

class LearningRateStepDecay(AbstractLearningRate):

    def __init__(self, lr0 = LR0, step_decay = STEP_DECAY, num_steps = NUM_STEPS, warmup = LR_WARMUP):
        self.lr0 = lr0
        self.step_decay = step_decay
        self.num_steps = num_steps
        self.warmup = warmup
        self.name = 'step_decay'

        points = np.linspace(self.warmup, 1, self.num_steps + 1)
        self.points = np.concatenate([np.asarray([0.]), points])

    def calculate(self, p: float):
        
        for i in range(len(self.points) - 1):
            if p <= self.points[i + 1] + 1e-6:

                return self.lr0 / (self.step_decay ** i)

    @property
    def config(self):
        config = {
                    'lr0': self.lr0,
                    'step_decay': self.step_decay,
                    'num_steps': self.num_steps,
                    'warmup': self.warmup,
                    'points': self.points,
                    'step_decay': self.name
                }
        return config

class LearningRateLinear(AbstractLearningRate):

    def __init__(self, start: float = LR_START_LINEAR, stop: float = LR_STOP_LINEAR, warmup = LR_WARMUP):

        self.start = start
        self.stop = stop
        self.warmup = warmup
        self.name = 'linear'

        self.a = stop - start
        self.b = start

    def calculate(self, p: float):
        if p <= self.warmup:
            return self.a
        return self.a * (p - self.warmup) + self.b

    @property
    def config(self):
        config = {
                    'start': self.start,
                    'stop': self.stop,
                    'warmup': self.warmup,
                    'a': self.a,
                    'b': self.b,
                    'linear': self.name
                }
        return config

class LearningRateLog(AbstractLearningRate):

    def __init__(self, start: float = LR_START_LOG, stop: float = LR_STOP_LOG, warmup = LR_WARMUP):

        self.start = start
        self.stop = stop
        self.warmup = warmup
        self.name = 'log'

        self.a = stop - start
        self.b = start

    def calculate(self, p: float):
        if p <= self.warmup:
            return 10 ** self.a
        return 10 ** (self.a * (p - self.warmup) + self.b)

    @property
    def config(self):
        config = {
                    'start': self.start,
                    'stop': self.stop,
                    'warmup': self.warmup,
                    'a': self.a,
                    'b': self.b,
                    'name': self.name
                }
        return config


class LearningRateFactory:

    @staticmethod
    def get_function(name: str, **kwargs):

        if name == 'constant':
            return LearningRateConstant(**kwargs)

        elif name == 'exp_decay':
            return LearningRateExpDecay(**kwargs)

        elif name == 'step_decay':
            return LearningRateStepDecay(**kwargs)

        elif name == 'linear':
            return LearningRateLinear(**kwargs)

        elif name == 'log':
            return LearningRateLog(**kwargs)
