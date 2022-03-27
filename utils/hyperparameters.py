import numpy as np

from config import *


LR_WARMUP = 0.4
LAMBDA_WARMUP = 0.2
LR0 = 5e-4
ALPHA = 5.
BETA = 0.75
GAMMA = 10.

def learning_rate_decay(p: float):
	if p <= LR_WARMUP:
		return LR0
	return LR0 / ((1 + ALPHA * (p - LR_WARMUP)) ** BETA)

def lambda_grl(p: float):
	if p <= LAMBDA_WARMUP:
		return 0.
	return (2 / (1 + np.exp(- GAMMA * (p - LAMBDA_WARMUP)))) - 1