import numpy as np

from config import *


LR_WARMUP = 0.4
LAMBDA_WARMUP = 0.05
LR0 = 5e-4
ALPHA = 5.
BETA = 0.75
GAMMA = 7.

STEP_DECAY = 2.
NUM_STEPS = 3

LR_START_LOG = -4.
LR_STOP_LOG = -2.

LR_START_LINEAR = 10 ** LR_START_LOG
LR_STOP_LINEAR = 10 ** LR_STOP_LOG


class LambdaGradientReversalLayer():
	
	def __init__(self, warmup = LAMBDA_WARMUP, gamma = GAMMA):
		self.warmup = warmup
		self.gamma = gamma

	def calculate(self, p: float):
		if p <= self.warmup:
			return 0.
		return (2 / (1 + np.exp(- self.gamma * (p - self.warmup)))) - 1


def lambda_grl(p: float):
	if p <= LAMBDA_WARMUP:
		return 0.
	return (2 / (1 + np.exp(- GAMMA * (p - LAMBDA_WARMUP)))) - 1
	# return 0.