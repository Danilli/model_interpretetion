import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from timeit import default_timer as timer
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import time




