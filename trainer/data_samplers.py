import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from IPython.core.debugger import Pdb
import pickle


class DataSampler:
    def __init__(self,lengths,batch_size, shuffle= True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.randomize()


    def randomize(self):
        N = len(self.lengths)
        self.ind = np.arange(0,len(self.lengths))
        if self.shuffle:
            np.random.shuffle(self.ind)
        self.ind = list(self.ind)
        self.ind.sort(key = lambda x: self.lengths[x],reverse=True)
        if self.shuffle:
            self.block_ids = {}
            random_block_ids = list(range(N))
            np.random.shuffle(random_block_ids)
            #generate a random number between 0 to N - 1
            blockid = random_block_ids[0]
            self.block_ids[self.ind[0]] = blockid
            running_count = 1
            for ind_it in range(1,N):
                if running_count >= self.batch_size or self.lengths[self.ind[ind_it]] != self.lengths[self.ind[ind_it-1]]:
                    blockid = random_block_ids[ind_it]
                    running_count = 0
                #   
                self.block_ids[self.ind[ind_it]] = blockid
                running_count += 1
            #  
            # Pdb().set_trace()
            self.ind.sort(key = lambda x: self.block_ids[x])


    def __iter__(self):
        # Pdb().set_trace()
        if self.shuffle:
            self.randomize()
        #
        return iter(self.ind)

    def __len__(self):
        return len(self.ind)




class BatchSampler:
    def __init__(self, lengths, batch_size,drop_last = False, shuffle=True, max_words_batch = 4000000,indices=None):
        self.lengths = lengths
        self.sampler = DataSampler(self.lengths,batch_size, shuffle=shuffle)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.max_words_batch = max_words_batch
        if indices is None:
            self.indices = list(range(lengths))
        else:
            self.indices = indices
    def __iter__(self):
        batch = []
        prev_len = -1
        this_batch_counter = 0
        for idx in  self.sampler:
            #if self.data_source.examples[idx][4] == self.unk_emb:
            #    continue
            #
            curr_len = self.lengths[idx]
            if ((prev_len > 0) and ((curr_len != prev_len) or (prev_len*len(batch) >= self.max_words_batch))):
                yield batch
                batch = []
                this_batch_counter = 0
            #
            batch.append(self.indices[idx])
            prev_len = curr_len
            this_batch_counter += 1
            if this_batch_counter == self.batch_size:
                yield batch
                batch = []
                prev_len = -1
                this_batch_counter = 0
        #
        if len(batch) > 0 and not self.drop_last:
            yield batch
            #self.sampler.randomize()
            prev_len = -1
            this_batch_counter = 0


    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
