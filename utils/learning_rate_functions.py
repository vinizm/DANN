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

    def __init__(self, **kwargs):
        self.const = kwargs.get('const', LR0)

    def calculate(self, p: float):
        return self.const

    @property
    def config(self):
        config = {'const': self.const}
        return config

class LearningRateExpDecay(AbstractLearningRate):

    def __init__(self, **kwargs):
        self.lr0 = kwargs.get('lr0', LR0)
        self.warmup = kwargs.get('warmup', LR_WARMUP)
        self.alpha = kwargs.get('alpha', ALPHA)
        self.beta = kwargs.get('beta', BETA)
        self.name = 'exp_decay'

    def calculate(self, p: float):
        if p <= self.warmup:
            return self.lr0
        
        u = (p - self.warmup) / (1 - self.warmup)
        return self.lr0 / ((1 + self.alpha * u) ** self.beta)

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

    def __init__(self, **kwargs):
        self.lr0 = kwargs.get('lr0', LR0)
        self.step_decay = kwargs.get('step_decay', STEP_DECAY)
        self.num_steps = kwargs.get('num_steps', NUM_STEPS)
        self.warmup = kwargs.get('warmup', LR_WARMUP)
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

    def __init__(self, **kwargs):
        
        self.start = kwargs.get('start', LR_START_LINEAR)
        self.stop = kwargs.get('stop', LR_STOP_LINEAR)
        self.warmup = kwargs.get('warmup', LR_WARMUP)
        self.name = 'linear'

        self.a = self.stop - self.start
        self.b = self.start

    def calculate(self, p: float):
        if p <= self.warmup:
            return self.b
        
        u = (p - self.warmup) / (1 - self.warmup)
        return self.a * u + self.b

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

    def __init__(self, **kwargs):

        self.start = kwargs.get('start', LR_START_LOG)
        self.stop = kwargs.get('stop', LR_STOP_LOG)
        self.warmup = kwargs.get('warmup', LR_WARMUP)
        self.name = 'log'

        self.a = self.stop - self.start
        self.b = self.start

    def calculate(self, p: float):
        if p <= self.warmup:
            return 10 ** self.b
        
        u = (p - self.warmup) / (1 - self.warmup)
        return 10 ** (self.a * u + self.b)

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


def LearningRateFactory(name: str, **kwargs):

    if name == 'constant':
        return LearningRateConstant(**kwargs)

    elif name == 'exp':
        return LearningRateExpDecay(**kwargs)

    elif name == 'step':
        return LearningRateStepDecay(**kwargs)

    elif name == 'linear':
        return LearningRateLinear(**kwargs)

    elif name == 'log':
        return LearningRateLog(**kwargs)