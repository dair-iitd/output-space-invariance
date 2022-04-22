from torch import nn
import torch
from IPython.core.debugger import Pdb 
import pickle
import os
import torch.nn.functional as F
import math
import time

from jacinle.logging import get_logger, set_output_file

logger = get_logger(__file__)

class LayerNormLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                 + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


