import torch

# Device
MY_DEVICE = 'cpu'
USE_CUDA = torch.cuda.is_available()

# Network configuration
NUM_NEURONS_IN = 1
NUM_NEURONS_OUT = 1

# Simulation timing
SIMULATION_WINDOW = 500
N = 10000

# SRM model constants
THRESHOLD = 1.0
TAU_C = 1.2
TAU_M = 20.0
A = 1000.0
ALPHA = 1.5
BETA = 1.0

# Gradient descent
MU = 0.01
ETA_RESET = 5

# Training
N_TRIALS = 100
SAVE_RESULTS = True
SAVE_PATH = "results/"

# Loss
TAU = 150.0

# Verbosity
VERBOSE = True
