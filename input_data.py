from collections import namedtuple
import torch

# Auto Encoder
learning_rate = 0.001
Encoder_hidden1 = 128
Encoder_hidden2 = 2
Decoder_hidden3 = 8
#FFN
FFN_hidden1 = 128
FFN_hidden2 = 256
FFN_out = 2
#Q_net
q_hidden1 = 128
q_hidden2 = 64
q_hidden3 = 32
q_out = 1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
State = namedtuple('State', ('partial_solution', 'partial_solution_nodes', 'neighbors', 'length'))
SEED = 1  # A seed for the random number generator
INIT_LR = 1e-3
MIN_EPSILON = 0.1
EPSILON_DECAY_RATE = 6e-5  # epsilon decay
FOLDER_NAME = './models'  # where to checkpoint the models