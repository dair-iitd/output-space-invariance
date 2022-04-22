#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plot.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/19/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

import io as _io
import numpy as np

from jacinle.image.backend import cv2, Image, opencv_only, pil_only

__all__ = ['plot2opencv', 'plot2pil']


@opencv_only
def plot2opencv(plt):
    """Convert a pyplot instance to image"""

    buf = _io.BytesIO()
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    rawbuf = np.frombuffer(buf.getvalue(), dtype='uint8')
    im = cv2.imdecode(rawbuf, cv2.IMREAD_COLOR)
    buf.close()
    return im


@pil_only
def plot2pil(plt):
    buffer = _io.StringIO()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil = Image.fromstring('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    return pil

