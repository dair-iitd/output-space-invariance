#! /bin/bash
#
# _jac-init.sh
# Copyright (C) 2019 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#

JACROOT=$1

export PYTHONPATH=$JACROOT:./:$PYTHONPATH
fname=`python $JACROOT/bin/_jac-init-gen.py`
source $fname
rm -f $fname

