from abc import ABC, abstractmethod

from utils.hyperparameters import LR_WARMUP, LR0, ALPHA, BETA


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

class LearningRateDecay(AbstractLearningRate):

    def __init__(self, lr0 = LR0, lr_warmup = LR_WARMUP, alpha = ALPHA, beta = BETA):
        self.lr0 = lr0
        self.lr_warmup = lr_warmup
        self.alpha = alpha
        self.beta = beta

    def calculate(self, p: float):
        if p <= self.lr_warmup:
            return self.lr0
        return self.lr0 / ((1 + self.alpha * (p - self.lr_warmup)) ** self.beta)

