import math
import pickle
from IPython.core.debugger import Pdb
import copy
import collections
import functools
import os
import json
from collections import Counter 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import random as py_random
import jacinle.random as random
import jacinle.io as io
import jactorch.nn as jacnn

from jactorch.utils.meta import as_tensor, as_float, as_cpu
from difflogic.cli import format_args
from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.data.dataloader import JacDataLoader
from jactorch.optim.accum_grad import AccumGrad
from jactorch.train.env import TrainerEnv
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor
import time
import datetime


import dgl
from .gcp_rrcn import GCPRRCN
from .gcp_bin_rrcn import GCP_BIN_RRCN
from .futoshiki_rrcn import FutoshikiRRCN
from .futo_bin_rrcn import Futoshiki_BIN_RRCN
from .nlm_models import NLMModel 
import utils

logger = get_logger(__file__)

def get_model(args):
    if args.model == 'nlm':
        rmodel = NLMModel(args) 
    elif args.model == 'gcp':
        if args.binary_model:
            logger.info("Creating a binary GCP model")
            rmodel = GCP_BIN_RRCN(args) 
        else:
            logger.info("Creating a MV GCP model")
            rmodel = GCPRRCN(args) 
    elif args.model == 'futo':
        if args.binary_model:
            logger.info("Creating a binary FUTO model")
            rmodel = Futoshiki_BIN_RRCN(args) 
        else:
            logger.info("Creating a MV FUTO model")
            rmodel = FutoshikiRRCN(args) 
    # 
    if args.use_gpu:
        rmodel = rmodel.cuda()
    return rmodel

