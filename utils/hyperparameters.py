import numpy as np

from config import *


LR_WARMUP = 0.4
LAMBDA_WARMUP = 0.2
LR0 = 5e-4
ALPHA = 5.
BETA = 0.75
GAMMA = 10.

STEP_DECAY = 2.
NUM_STEPS = 3

LR_START_LOG = -4.
LR_STOP_LOG = -2.

LR_START_LINEAR = 10 ** LR_START_LOG
LR_STOP_LINEAR = 10 ** LR_STOP_LOG


def lambda_grl(p: float):
	# if p <= LAMBDA_WARMUP:
	# 	return 0.
	# return (2 / (1 + np.exp(- GAMMA * (p - LAMBDA_WARMUP)))) - 1
	return 0.