from chainer.dataset import iterator


class PrefetchMultiprocessIterator(iterator.Iterator):
    def __init__(self, dataset, batch_size, repeat=True, shuffle=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle

    def __next__(self):
        pass

    @property
    def current_position(self):
        pass

    @property
    def epoch(self):
        pass

    @property
    def is_new_epoch(self):
        pass

    @property
    def epoch_detail(self):
        pass

    @property
    def previous_epoch_detail(self):
        pass
