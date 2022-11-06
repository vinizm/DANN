import numpy as np

from config import *


LR_WARMUP = 0.
LAMBDA_WARMUP = 0.
LR0 = 5e-4
ALPHA = 2.25
BETA = 0.75
GAMMA = 10.

STEP_DECAY = 2.
NUM_STEPS = 3

LR_START_LOG = -2.
LR_STOP_LOG = -4.

LR_START_LINEAR = 10 ** LR_START_LOG
LR_STOP_LINEAR = 10 ** LR_STOP_LOG

LAMBDA_SCALE = 1.


class LambdaGradientReversalLayer():
	
	def __init__(self, warmup = LAMBDA_WARMUP, gamma = GAMMA, lambda_scale = LAMBDA_SCALE):
		self.warmup = warmup
		self.gamma = gamma
		self.lambda_scale = lambda_scale

	def calculate(self, p: float):
		if p <= self.warmup:
			return 0.

		u = (p - self.warmup) / (1 - self.warmup)
		return self.lambda_scale * ((2 / (1 + np.exp(- self.gamma * u))) - 1)

	@property
	def config(self):
		config = {
					'warmup': self.warmup,
					'gamma': self.gamma,
					'lambda_scale': self.lambda_scale
				}
		return config