#! /usr/bin/env python3
import sys

from os.path import expanduser
import models.models as models
import logging
import scheduler
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
import dgl

from jactorch.utils.meta import as_tensor, as_float, as_cpu
from difflogic.cli import format_args
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference
from dataset.dataset import FutoshikiDataset, GCPDataset, NLMDataset

from difflogic.thutils_rl import binary_accuracy, instance_accuracy
from difflogic.train import TrainerBase
import utils

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.data.dataloader import JacDataLoader
from jactorch.optim.quickaccess import get_optimizer
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor
import time
import datetime
import warnings
from torch.distributions.categorical import Categorical
#import data_sampler
import data_samplers
warnings.simplefilter('once')
torch.set_printoptions(linewidth=150)
torch.set_num_threads(4)

TASKS = [
    'futoshiki', 'sudoku', 'gcp']

parser = JacArgumentParser()
parser.add_argument('--num-workers', type=int, default = 0, help = 'num workers for data loader')
parser.add_argument('--upper-limit-on-grad-norm', type=float, default=1000, help= ' if grad is beyond this number, then ignore that update step')

parser.add_argument(
    '--model',
    default='gcp',
    choices=['edge.futo','gcp','futo','nlm'],
    help='model choices.')


nlm_group = parser.add_argument_group('NLM model specific args')
LogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 30,
        'breadth': 2,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')

nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)


rrn_group = parser.add_argument_group('rrn model specific args')

rrn_group.add_argument('--max-num-embeddings', type=int,
                    default=9, metavar='N', help='how many embeddings should the embed generator  produce?')

rrn_group.add_argument('--embedding-ortho-loss-factor', type=float,
                    default=0, metavar='N', help='add component to loss to ensure initialised digit embeddings are orthogonal')

rrn_group.add_argument('--msg-passing-steps', type=int,
                       default=32, metavar='N', help='num msg passing steps')

rrn_group.add_argument('--layer-norm', type=int,
                       default=1, metavar='N', help='should layer normalize lstm?')

rrn_group.add_argument('--hidden-dim', type=int,
                       default=96, metavar='N', help='hidden dim')

rrn_group.add_argument('--attn-dim', type=int,
                       default=96, metavar='N', help='attn  dim for computing attn across msgs')

rrn_group.add_argument('--attention-edges', nargs='*',
                       default=[], help='edges over which attn should be applied')

rrn_group.add_argument('--edge-dropout', type=float, default=0.1,
                       metavar='N', help='dropout for msg passing')

rrn_group.add_argument('--binary-model',type = int, default = 0, help='make a binary version of the pipeline')

rrn_group.add_argument('--logk',type = int, default = 0, help='make logk model (GCP)')

rrn_group.add_argument('--edges-within-cluster-nodes',type = int, default = 0, help='add edges within cluster nodes?')

rrn_group.add_argument('--edge-embeddings-dim',type =int, default = 0, help='should we have edge features?')

rrn_group.add_argument('--share-all-msg-passing-mlp',type =int, default = 0, help='should all the msg passing wts be shared across edges?')
rrn_group.add_argument('--share-lt-edges',type = int, default = 0, help='share inter and intra level lt edges for Futoshiki in binary graph')

rrn_group.add_argument('--share-mp-wts-in-gen',type =int, default = 1, help='should the msg passing wts be shared?')
rrn_group.add_argument('--share-diff-edges-in-gen',type =int, default = 1, help='should share the msg passing wts for diff edges b/w generator and rrn?')

rrn_group.add_argument('--output-embedding-generator',type = str, default = 'eye', choices=['eye','linear'], help='relationship between input and output embeddings')


rrn_group.add_argument('--diff-edges-in-class-graph',type = int, default =1, help='have diff edges in class graph?')
rrn_group.add_argument('--class-rrn-num-steps',type = int, default =5, help='number of steps of msg passing for class graph?')


# task related
task_group = parser.add_argument_group('Task')
task_group.add_argument(
    '--task', required=True, choices=TASKS, help='tasks choices')

data_gen_group = parser.add_argument_group('Dataset creation')

data_gen_group.add_argument('--train-data-size', type=int, default=-1,
                            metavar='M', help='size of training data ')


data_gen_group.add_argument(
    '--train-file', type=str, help="train data file", default='')

data_gen_group.add_argument('--test-file', type=str, help="test data file")

train_group = parser.add_argument_group('Train')

train_group.add_argument('--min-loss', type=int, default=0,
                            help='compute minimum of loss over possible solutions')

train_group.add_argument(
    '--start-epoch',
    type=int,
    default=1,
    help='use when starting from a checkpoint')

train_group.add_argument(
    '--incomplete-targetset',
    type=int,
    default=0,
    metavar='MISSING TARGETS',
    help='is target set incomplete? Used in GCP task')

train_group.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='SEED',
    help='seed of jacinle.random')

train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')

train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW','Adagrad','Adadelta'],
    help='optimizer choices')

train_group.add_argument(
    '--lr',
    type=float,
    default=0.0002,
    metavar='F',
    help='initial learning rate')

train_group.add_argument(
    '--wt-decay',
    type=float,
    default=0.0001,
    metavar='F',
    help='weight decay of learning rate per lesson')

train_group.add_argument(
    '--grad-clip',
    type=float,
    default=5.0,
    metavar='F',
    help='value at which gradients need to be clipped')

train_group.add_argument(
    '--batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for training')

train_group.add_argument(
    '--controlled-batching',
    type=int,
    default=0,
    metavar='N',
    help='Controlled batching? Ensuring same number of nodes in a batch')


train_group.add_argument(
    '--test-batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for testing')

train_group.add_argument(
    '--reduce-lr',
    type=int,
    default=0,
    metavar='N',
    help='should reduce lr?')


# Note that nr_examples_per_epoch = epoch_size * batch_size
TrainerBase.make_trainer_parser(
    parser, {
        'epochs': 100,
        'epoch_size': 250,
    })

train_group.add_argument('--should-stop-training-on-flattening', type=int,
                         default=0, help="should we stop training when scheduler says so ?")


io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', type=str, default=None, metavar='DIR', help='dump dir')
io_group.add_argument(
    '--load-checkpoint',
    type=str,
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')

schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=200,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')

schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=None,
    metavar='N',
    help='the interval(number of epochs) to do test')

schedule_group.add_argument(
    '--test-begin-epoch',
    type=int,
    default=0,
    metavar='N',
    help='the interval(number of epochs) after which test starts')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')

schedule_group.add_argument(
    '--analysis-only', action='store_true', help='analysis-only mode. return the trainer object')

schedule_group.add_argument(
    '--continue-train', type=int,default=1, help='Continue training from the last/latest checkpoint available')


logger = get_logger(__file__)

glogger = logging.getLogger("grad")
glogger.setLevel(logging.INFO)
blogger = logging.getLogger("batchinfo")
blogger.setLevel(logging.INFO)


def setup(arg_str=None):
    global args
    global glogger
    global blogger
    global logger
    #sys.path.insert(0,'third_party/Jacinle/bin')
    if arg_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_str.split())
    
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    
    if args.dump_dir is not None:
        io.mkdir(args.dump_dir)
        args.log_file = os.path.join(args.dump_dir, 'log.log')
        set_output_file(args.log_file)
    
        grad_handle = logging.FileHandler(os.path.join(args.dump_dir, 'grad.csv'))
        glogger.addHandler(grad_handle)
        glogger.propagate = False
        glogger.info(
            'epoch,iter,loss,grad_norm_before_clip,grad_norm_after_clip,param_norm_before_clip')
    
        batch_handle = logging.FileHandler(os.path.join(args.dump_dir, 'batch_info.csv'))
        blogger.addHandler(batch_handle)
        blogger.propagate = False
        blogger_header ='epoch,iter,' + ','.join(['e{}'.format(i) for i in range(args.batch_size)])
        blogger.info(blogger_header)
    
    else:
        args.checkpoints_dir = None
        args.summary_file = None
    
    if args.seed is not None:
        import jacinle.random as random
        random.reset_global_seed(args.seed)
        utils.setup_seed(args.seed,args.use_gpu) 
    
    args.task_is_futoshiki = args.task in ['futoshiki']
    args.task_is_sudoku = args.task in ['sudoku']
    args.task_is_gcp = args.task in ['gcp']
    return args


def make_dataset(epoch_size, is_train):
    if args.task_is_futoshiki:
        data_file = args.train_file
        if not is_train:
            data_file = args.test_file
        if args.model == 'nlm':
           return NLMDataset(epoch_size = epoch_size,
                 data_size=args.train_data_size if is_train else -1,
                train_dev_test=0 if is_train else 2,
                data_file=data_file,
                task = args.task,
                args=args)
        else: 
            return FutoshikiDataset(epoch_size=epoch_size,
                             data_size=args.train_data_size if is_train else -1,
                             train_dev_test=0 if is_train else 2,
                             data_file=data_file,
                             args=args
                             )
    elif (args.task_is_gcp or args.task_is_sudoku):
        data_file = args.train_file
        if not is_train:
            data_file = args.test_file
        
        if args.model == 'nlm':
           return NLMDataset(epoch_size = epoch_size,
                 data_size=args.train_data_size if is_train else -1,
                train_dev_test=0 if is_train else 2,
                data_file=data_file,
                task = args.task,
                args=args)
        else:
            return GCPDataset(epoch_size=epoch_size,
                             data_size=args.train_data_size if is_train else -1,
                             train_dev_test=0 if is_train else 2,
                             data_file=data_file,
                             args=args
                             )

    else:
        raise

def default_reduce_func(k, v):
    return v.mean()


class MyTrainer(TrainerBase):
    def reset_test(self):
        if 'test' in self.datasets:
            del self.datasets['test']
        if 'test' in self.data_iterator:
            del self.data_iterator['test'] 
        self.querywise_result = []
        self.args.get_stepwise_accuracy = True
        self.extra_args_for_test_at_end = {'get_all_steps': True}

    def step(self, feed_dict, reduce_func=default_reduce_func, cast_tensor=False):
        assert self._model.training, 'Step a evaluation-mode model.'
        self.num_iters += 1
        if cast_tensor:
            feed_dict = as_tensor(feed_dict)

        begin = time.time()
        loss, monitors, output_dict = self._model(feed_dict)
        
        t0 = time.time()
        time_forward = t0 - begin 

        loss = reduce_func('loss', loss)
        loss_f = as_float(loss)

        monitors = {k: reduce_func(k, v) for k, v in monitors.items()}
       
        monitors_f = as_float(monitors)

        self._optimizer.zero_grad()
        
        t0 = time.time()
        if loss.requires_grad:
            loss.backward()
        
        t1 = time.time()
        time_backprop = t1 - t0 

        grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip =  0, 0, 0 
        if loss.requires_grad:
            grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip = utils.gradient_normalization(
                self._model, grad_norm=args.grad_clip)
            if grad_norm_before_clip <= args.upper_limit_on_grad_norm:
                self._optimizer.step()
            else:
                self.num_bad_updates += 1
                logger.info('not taking optim step. Grad too high {}. Num bad updates: {}'.format(round(grad_norm_before_clip,2), self.num_bad_updates))
        
        glogger.info(','.join(map(lambda x: str(round(x, 6)), [self.current_epoch, self.num_iters, loss_f, grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip])))
        current_indices = ','.join(map(str,feed_dict['qid'].squeeze(-1).cpu().numpy()))
        blogger.info('{},{},{}'.format(self.current_epoch,self.num_iters,current_indices))
        end = time.time()

        return loss_f, monitors_f, output_dict, {'time_forward': time_forward, 'time_backprop': time_backprop, 'time/gpu': end - begin}

    def save_checkpoint(self, name, epoch=None):

        if args.checkpoints_dir is not None:
            checkpoint_file = os.path.join(args.checkpoints_dir,
                                           'checkpoint_{}.pth'.format(name))

            model = self._model
            
            if isinstance(model, nn.DataParallel):
                model = model.module

            state = {
                'model': as_cpu(model.state_dict()),
                'optimizer': as_cpu(self._optimizer.state_dict()),
                'extra': {'name': name, 'epoch': epoch}
            }
            
            try:
                torch.save(state, checkpoint_file)
                logger.info('Checkpoint saved: "{}".'.format(checkpoint_file))
            except Exception:
                logger.exception(
                    'Error occurred when dump checkpoint "{}".'.format(checkpoint_file))

    def load_checkpoint(self, filename):
        checkpoint = None
        if os.path.isfile(filename):
            model = self._model

            if isinstance(model, nn.DataParallel):
                model = model.module
            try:
                checkpoint = torch.load(filename)
                model.load_state_dict(checkpoint['model'],strict=False)
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                if args.use_gpu:
                    model.cuda()
                return checkpoint['extra']
            except Exception:
                logger.exception(
                    'Error occurred when load checkpoint "{}".'.format(filename))
                    
                if 'gcp_solver.embed_module.init_generator.fixed_embeddings' in checkpoint['model']:
                    num_emb = model.gcp_solver.embed_module.init_generator.fixed_embeddings.data.size(0)
                    num_existing_emb = checkpoint['model']['gcp_solver.embed_module.init_generator.fixed_embeddings'].size(0)
                    mean_before = model.gcp_solver.embed_module.init_generator.fixed_embeddings.mean(dim=1)
                    model.gcp_solver.embed_module.init_generator.fixed_embeddings.data[:num_existing_emb,:].copy_(checkpoint['model']['gcp_solver.embed_module.init_generator.fixed_embeddings'])
                    #model.gcp_solver.embed_module.init_generator.fixed_embeddings.shape
                    mean_after =  model.gcp_solver.embed_module.init_generator.fixed_embeddings.mean(dim=1)
                    print("mean before: ", mean_before)
                    print("mean after: ", mean_after)
                    logger.warning("Found only {} embeddings out of {}".format(num_existing_emb,num_emb))

        else:
            logger.warning(
                'No checkpoint found at specified position: "{}".'.format(filename))
        
        if checkpoint is not None:
            return checkpoint['extra']
        return None

    def _dump_meters(self, meters, mode):
        if args.summary_file is not None:
            meters_kv = meters._canonize_values('avg')
            
            meters_kv['mode'] = mode
            meters_kv['time'] = time.time()
            meters_kv['htime'] = str(datetime.datetime.now())
            meters_kv['config'] = args.dump_dir
            meters_kv['lr'] = self._optimizer.param_groups[0]['lr']
            if mode == 'train':
                meters_kv['epoch'] = self.current_epoch
                meters_kv['data_file'] = args.train_file
            else:
                sum_meters = meters._canonize_values('sum')
                meters_kv['runtime'] = sum_meters['runtime']
                meters_kv['test-batch-size'] = args.test_batch_size
                meters_kv['test-size'] = int(args.test_batch_size*sum_meters['number']/meters_kv['number'])
                meters_kv['mem-used'] = torch.cuda.max_memory_allocated()
                meters_kv['epoch'] = -1
                meters_kv['data_file'] = args.test_file
            with open(args.summary_file, 'a') as f:
                f.write(io.dumps_json(meters_kv))
                f.write('\n')

    data_iterator = {}
    datasets = {}

    def _prepare_dataset(self, epoch_size, mode):
        assert mode in ['train', 'test']

        if mode == 'train':
            batch_size = args.batch_size
        else:
            batch_size = args.test_batch_size
        
        # The actual number of instances in an epoch is epoch_size * batch_size.
        if mode in self.datasets:
            dataset = self.datasets[mode]
        else:
            dataset = make_dataset(epoch_size *
                                   batch_size, mode == 'train')
            self.datasets[mode] = dataset

        
        if self.args.controlled_batching:
            indices, lengths = self.datasets[mode].sample_epoch_indices()
            #Pdb().set_trace()
            batch_sampler = data_samplers.BatchSampler(lengths,batch_size = batch_size, shuffle=(mode=='train'),indices=indices)
            #batch_sampler = data_samplers.BatchSampler(lengths,batch_size = batch_size, shuffle=False,indices=indices)
            dataloader = JacDataLoader(dataset,
                                batch_sampler = batch_sampler, num_workers= min(epoch_size, args.num_workers), 
                                collate_fn = lambda x: utils.collate_graph(x,args)) 
        else:
            dataloader = JacDataLoader(
                dataset,
                shuffle=(mode == 'train'),
                batch_size=batch_size,
                num_workers=min(epoch_size, args.num_workers),
                collate_fn = lambda x: utils.collate_graph(x,args))

        self.data_iterator[mode] = dataloader.__iter__()

    def _get_data(self, index, meters, mode):
        feed_dict = self.data_iterator[mode].next()
        meters.update(number=feed_dict['n'].data.numpy().mean())
        return self._convert_to_cuda(feed_dict)

    def _convert_to_cuda(self,feed_dict):
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)
            for key in ['bg','bcg']:
                if key  in feed_dict and isinstance(feed_dict[key], dgl.BatchedDGLHeteroGraph):
                    feed_dict[key] = feed_dict[key].to(feed_dict['n'].device)
        return feed_dict


    #used in _test
    def _get_result(self, index, meters, mode):
#         Pdb().set_trace()
        torch.cuda.reset_max_memory_allocated()
        
        feed_dict = self._get_data(index, meters, mode)
        
        tic = time.time()
        if (self.test_at_end or self.args.analysis_only) and 'extra_args_for_test_at_end' in self.__dict__: 
            output_dict = self.model(feed_dict, **self.extra_args_for_test_at_end)
        else:
            output_dict = self.model(feed_dict)
        toc = time.time()
        
         
        if not isinstance(output_dict,dict):
            output_dict = output_dict[2]
        
        target = feed_dict['target']
        
        result, ia_output_dict = instance_accuracy(
            target, output_dict['pred'], return_float=True, feed_dict=feed_dict, task=args.task, args=args)
        
        succ = result['accuracy'] == 1.0
        if 'query_info' in ia_output_dict and (self.test_at_end or self.args.analysis_only) and 'querywise_result' in self.__dict__:
            this_result = [ia_output_dict['query_info'], ia_output_dict['querywise_accuracy']]
            if True:
            #if self.args.analysis_only:
                #this_result.append(ia_output_dict)
                #this_result.append(output_dict['pred'][:,:,:,-1].cpu())
                if 'bg' in feed_dict:
                    del feed_dict['bg']
                if 'bcg' in feed_dict:
                    del feed_dict['bcg']
                feed_dict = as_cpu(feed_dict)
                #feed_dict['pred'] = output_dict['pred'][:,:,:,-1].cpu()
                feed_dict['pred'] = output_dict['pred'].cpu().argmax(dim=1)
                this_result.append({'qid': feed_dict.get('qid',-1), 'pred': feed_dict['pred'],'k': feed_dict.get('chromatic_num',-1)})
                #Pdb().set_trace()
            self.querywise_result.append(this_result)
        
        meters.update(succ=succ)
        meters.update(result, n=target.size(0))
        meters.update(runtime=toc-tic)
#         Pdb().set_trace()
        message = '> {} iter={iter}, accuracy={accuracy:.4f}, pw={pointwise accuracy:.4f}'.format(
            mode, iter=index, **meters.val)
#         Pdb().set_trace()

        return message, dict(succ=succ, feed_dict=feed_dict)

    def _get_train_data(self, index, meters):
        return self._get_data(index, meters, mode='train')

    def _train_epoch(self, epoch_size, is_last=False):
        meters = super()._train_epoch(epoch_size)
        logger.info("Best Dev Accuracy: {}".format(self.best_accuracy))
        i = self.current_epoch

        if args.save_interval is not None and i % args.save_interval == 0:
            self.save_checkpoint(str(i))
        self.save_checkpoint('last', i)
        

        test_meters = None
        if (is_last or (args.test_interval is not None and i % args.test_interval == 0 and i > args.test_begin_epoch)):
            #self.reset_test()
            test_meters = self.test()
            if self.best_accuracy < test_meters[0].avg["corrected accuracy"]:
                self.best_accuracy = test_meters[0].avg["corrected accuracy"]
                self.save_checkpoint("best")
        return meters, test_meters

    def train(self, start_epoch=1, num_epochs=0):
        self.early_stopped = False
        meters = None

        for i in range(start_epoch, start_epoch + num_epochs):
            self.current_epoch = i
            meters, test_meters = self._train_epoch(
                self.epoch_size, (self.current_epoch == (start_epoch + num_epochs-1)))

            if args.reduce_lr and test_meters is not None:
                metric = test_meters[0].avg["corrected accuracy"]
                self.my_lr_scheduler.step(1.0-1.0*metric)
                if args.should_stop_training_on_flattening and self.my_lr_scheduler.shouldStopTraining():
                    logger.info("Stop training as no improvement in accuracy - no of unconstrainedBadEopchs: {0} > {1}".format(
                        self.my_lr_scheduler.unconstrainedBadEpochs, self.my_lr_scheduler.maxPatienceToStopTraining))
                    break

        return meters, test_meters



def test_at_end(trainer):
    logger.info("++++++++ START RUNNING TEST AT END -------")
    trainer.test_at_end = True
    test_files = {}
    #
    if args.task_is_sudoku:
        test_files = {
            
                'data/sudoku_test_bs-24-mask30to52_amb.pkl': 'bs24amb_test_30-52',
#                 'data/sudoku_test_bs-24-mask30to70_amb.pkl': 'bs24amb_test',
                'data/sudoku_test_bs-25-mask30to52_amb.pkl': 'bs25amb_test',
#                 'data/sudoku_train_bs-24-mask30to70_amb.pkl': 'bs24amb_train',
#                 'data/sudoku_train_bs-25-mask30to48_amb.pkl': 'bs25amb_train',
            
            
#                 'data/sudoku_test_bs-16-mask30to58_amb.pkl': 'bs16amb',
#                 'data/sudoku_test_bs-15-mask30to58_amb.pkl': 'bs15amb',
#                 'data/sudoku_train_bs-15-mask30to58_mix.pkl': 'bs15mix_train',
            
#                 'data/sudoku_v2_test_bs-16-mask30to58_unq.pkl':'bs16unq_v2',
#                 'data/sudoku_v2_test_bs-15-mask30to58_unq.pkl':'bs15unq_v2',
#                 'data/sudoku_v2_test_bs-12-mask30to59_unq.pkl': 'bs12unq_v2',
#                 'data/sudoku_v2_test_bs-8-mask30to67_unq.pkl': 'bs8unq_v2',
#                 'data/sudoku_v2_test_bs-9-mask30to67_unq.pkl': 'bs9unq_v2',
#                 'data/sudoku_v2_test_bs-10-mask30to63_unq.pkl': 'bs10unq_v2',
               
#                 'data/sudoku_test_bs-16-mask30to70_unq.pkl':'bs16unq',
#                 'data/sudoku_test_bs-15-mask30to70_unq.pkl':'bs15unq',
#                 'data/sudoku_test_bs-10-mask30to70_unq.pkl': 'bs10unq',
#                 'data/sudoku_test_bs-12-mask30to70_unq.pkl': 'bs12unq',
#                 'data/sudoku_test_bs-14-mask30to70.pkl': 'bs14',
#                 'data/sudoku_9_all_unq_val.pkl': 'bs9val',
            
#                 'data/sudoku_9_all_unq_test.pkl': 'bs9test',
#                 'data/sudoku_test_bs-8-mask30to70_unq.pkl': 'bs8unq',
                #'data/sudoku_test_bs-16-mask30to70.pkl': 'bs16',
                #'data/sudoku_test_bs-8-mask30to70.pkl': 'bs8',
                #'data/sudoku_test_bs-10-mask30to70.pkl': 'bs10',
                #'data/sudoku_test_bs-12-mask30to70.pkl': 'bs12',
                #'data/sudoku_test_bs-15-mask30to70.pkl': 'bs15',
                #'data/sudoku_9_all_unq_train.pkl': 'bs9train',
                } 
        
    if args.task_is_gcp:
                        
        test_files = {  'data/gcp_test_k-7_n-40to80_mask-30to70.pkl': 'k7_test',
                        'data/gcp_test_k-4_n-40to150_mask-30to70.pkl': 'k4_test',
                        'data/gcp_test_k-5_n-40to150_mask-30to70.pkl': 'k5_test',
                        'data/gcp_test_k-6_n-40to120_mask-30to70.pkl': 'k6_test',
                    }
   
    if args.task_is_futoshiki:
        
        test_files = {
                    'data/futo_unq_test_bs-10_mask-30to70.pkl':'test_10',
                    'data/futo_unq_test_bs-11_mask-30to70.pkl':'test_11',
                    'data/futo_unq_test_bs-12_mask-30to70.pkl':'test_12',
                    'data/futo_unq_test_bs-6_mask-30to70.pkl':'test_6',
                    'data/futo_unq_test_bs-7_mask-30to70.pkl':'test_7',
                    'data/futo_unq_test_bs-8_mask-30to70.pkl':'test_8',
                    'data/futo_unq_test_bs-9_mask-30to70.pkl':'test_9',
                    'data/futo_unq_test_bs-5_mask-30to70.pkl':'test_5',
                    'data/futo_unq_val_bs-5_mask-30to70.pkl':'val_5',
                    'data/futo_unq_train_bs-5_mask-30to70.pkl':'train_5',
                    'data/futo_unq_val_bs-6_mask-30to70.pkl':'val_6',
                    'data/futo_unq_train_bs-6_mask-30to70.pkl':'train_6',
                }
        
    if not args.task_is_gcp:
        if args.train_file not in test_files:
            test_files[args.train_file] = 'train'

        if args.test_file not in test_files:
            test_files[args.test_file] = 'val'

    args.test_only = 1
    home = expanduser("~")
    for tf in test_files:
        #args.test_file = os.path.join(home,'hpcscratch/nlm',tf)
        args.test_file = tf 
        logger.info("Testing for: {}".format(args.test_file))
        trainer.reset_test()
        querywise_file = os.path.join(args.current_dump_dir, test_files[tf]+"_querywise.pkl")
        if not os.path.exists(querywise_file):
            rv = trainer.test()
            with open(querywise_file, 'wb') as fh:
                pickle.dump(trainer.querywise_result, file = fh)
        else:
            logger.info("Querywise file alredy exists: {}".format(querywise_file))
    # 
    trainer.test_at_end = False

def main():
    if args.dump_dir is not None:
        args.current_dump_dir = args.dump_dir

        args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')
        args.checkpoints_dir = os.path.join(
            args.current_dump_dir, 'checkpoints')
        io.mkdir(args.checkpoints_dir)

    exp_fh = open(os.path.join(args.current_dump_dir,'exp.sh'),'a')
    print('jac-run {}'.format(' '.join(sys.argv)),file=exp_fh)
    exp_fh.close()

    logger.info('jac-run {}'.format(' '.join(sys.argv))) 
    logger.info(format_args(args))
    model = models.get_model(args)

    if args.use_gpu:
        model.cuda()
    optimizer = get_optimizer(args.optimizer, model,
                              args.lr, weight_decay=args.wt_decay)


    trainer = MyTrainer.from_args(model, optimizer, args)
    trainer.args = args
    trainer.test_at_end = False
    trainer.num_iters = 0
    trainer.num_bad_updates = 0
    trainer.test_batch_size = args.test_batch_size
    
    start_epoch = args.start_epoch
    epochs = args.epochs
    #Pdb().set_trace()
    if (args.load_checkpoint is None) and args.continue_train and (not args.test_only) and (not args.analysis_only):
        
        last_chkpt_path = os.path.join(args.checkpoints_dir,'checkpoint_last.pth')
        extras = torch.load(last_chkpt_path)['extra'] if os.path.isfile(last_chkpt_path) else {}
        if ('epoch' in extras) and (extras['epoch'] is not None):
            start_epoch = extras['epoch']+1
            args.load_checkpoint = last_chkpt_path
        else:
            max_epoch = 0
            for f_name in os.listdir(args.checkpoints_dir):
                epoch_num = f_name.split('_')[1][:-4]
                max_epoch = max(max_epoch, int(epoch_num)) if epoch_num.isnumeric() else max_epoch
            if max_epoch:
                start_epoch = max_epoch + 1 
                args.load_checkpoint = os.path.join(args.checkpoints_dir,'checkpoint_{}.pth'.format(max_epoch))
        #
        epochs = args.epochs - start_epoch + 1
    else:
        if (args.analysis_only or args.test_only) and args.load_checkpoint is None:
            args.load_checkpoint = os.path.join(args.checkpoints_dir,'checkpoint_best.pth')

    if args.load_checkpoint is not None:
#         Pdb().set_trace()
        extras = trainer.load_checkpoint(args.load_checkpoint)
        if ('epoch' in extras) and (extras['epoch'] is not None):
            start_epoch = extras['epoch'] + 1
            epochs = args.epochs - start_epoch + 1
            #args.load_checkpoint = last_chkpt_path
        else:
            epoch_num = os.path.basename(args.load_checkpoint).split('_')[-1][:-4]
            start_epoch = max(start_epoch, 1+int(epoch_num)) if epoch_num.isnumeric() else start_epoch
            epochs = args.epochs - start_epoch + 1

    logger.info("Start epoch: {}. Total epochs: {}. Checkpoint: {}".format(start_epoch, epochs, args.load_checkpoint))
            
    my_lr_scheduler = scheduler.CustomReduceLROnPlateau(trainer._optimizer, {'mode': 'min', 'factor': 0.2, 'patience': math.ceil(
        7/args.test_interval), 'verbose': True, 'threshold': 0.01, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 0.01*args.lr, 'eps': 0.0000001}, maxPatienceToStopTraining=math.ceil(20/args.test_interval))
    
    trainer.my_lr_scheduler = my_lr_scheduler
   
    if args.analysis_only:
        return trainer

    if args.test_only:
        trainer.reset_test()
        test_at_end(trainer)
        trainer.reset_test()
        #rv = trainer.test()
        return None

    # setup add to target set
    trainer.model.add_to_targetset = args.min_loss and args.incomplete_targetset
   
    #save the initialized model
    
    trainer.save_checkpoint('initialization_{}'.format(start_epoch))
    meters, test_meters = trainer.train(
        start_epoch, epochs)
    
#     trainer.save_checkpoint('last', args.epochs)

    trainer.load_checkpoint(os.path.join(
        args.checkpoints_dir, 'checkpoint_best.pth'))
    
    logger.info("Best Dev Accuracy: {}".format(trainer.best_accuracy))
    
    trainer.reset_test()
    ret = trainer.test()
    
    test_at_end(trainer)
    return ret


if __name__ == '__main__':
    setup()
    #Pdb().set_trace()
    main()

