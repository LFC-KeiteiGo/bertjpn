import pickle
import os
import gc
import random
import torch

from src.Preprocess import Preprocess


class Batch(object):
    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)

            src = torch.tensor(data['src'])

            labels = torch.tensor(data['labels'])
            segs = torch.tensor(data['segs'])
            mask = ~(src == 0)

            clss = torch.tensor(data['clss'])
            mask_cls = torch.tensor(data['mask_cls'])

            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src', src.to(device))
            setattr(self, 'labels', labels.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask', mask.to(device))

            if is_test:
                src_str = data['src_line']
                setattr(self, 'src_str', src_str)

    def __len__(self):
        return self.batch_size

    @staticmethod
    def _pad(data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data


class DataLoader:
    def __init__(self, data_path, input_length, batch_size,
                 device='cuda', shuffle=True, is_test=False):
        self.path = data_path if data_path[-1] is '/' else data_path+'/'
        self.input_len = input_length
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.data = []
        self.load_data()

    def __iter__(self):
        dataset_iter = (d for d in self.data)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def load_data(self):
        path_list = [self.path+x for x in os.listdir(self.path) if '.pickle' in x]
        for path in path_list:
            with open(path, 'rb') as f:
                self.data.append(pickle.load(f))

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(dataset=self.cur_dataset,
                            batch_size=self.batch_size,
                            device=self.device,
                            shuffle=self.shuffle,
                            is_test=self.is_test)

    def preprocess(self, data, n):
        temp = Preprocess(data, n)


class DataIterator:
    def __init__(self, dataset, batch_size,
                 device=None, is_test=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size

        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test

        self.sort_key = lambda x: len(x[1])

        self.iterations = 0
        self._iterations_this_epoch = 0

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 50):

            p_batch = sorted(buffer, key=lambda x: len(x[3]))
            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                yield b

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        if 'labels' in ex:
            labels = ex['labels']
        else:
            labels = ex['src_sent_labels']

        segs = ex['segs']
        if not self.args.use_interval:
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        if is_test:
            return src, labels, segs, clss, src_txt, tgt_txt
        else:
            return src, labels, segs, clss

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if (len(ex['src']) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if (ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch
