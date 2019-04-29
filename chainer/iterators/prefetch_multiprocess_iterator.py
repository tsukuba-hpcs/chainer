import threading

from chainer.dataset import iterator

_response_time = 0.1


class PrefetchMultiprocessIterator(iterator.Iterator):
    def __init__(self, dataset, batch_size, local_storage_base,
                 n_generate_id, n_prefetch_from_backend, n_prefetch_batch,
                 n_remove_example, repeat=True, shuffle=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.local_storage_base = local_storage_base
        self.repeat = repeat
        self.shuffle = shuffle
        self.n_generate_id = n_generate_id
        self.n_prefetch_from_backend = n_prefetch_from_backend
        self.n_prefetch_batch = n_prefetch_batch
        self.n_remove_example = n_remove_example

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


class _Communicator(object):
    STATUS_CONTINUE = 0
    STATUS_RESET = 1
    STATUS_TERMINATE = 2

    def __init__(self, n_prefetch, dataset_timeout):
        self.n_prefetch = n_prefetch
        self.dataset_timeout = dataset_timeout

        self._lock = threading.Lock()
        self._not_empty_cond = threading.Condition(self._lock)
        self._not_full_cond = threading.Condition(self._lock)
        self._batch_queue = []
        self._status = _Communicator.STATUS_CONTINUE
        self._reset_count = 0
        self._prefetch_state = None

    @property
    def is_terminated(self):
        with self._lock:
            return self._status == _Communicator.STATUS_TERMINATE

    # called from iterator
    def get(self):
        with self._lock:
            while not self._batch_queue:
                self._not_empty_cond.wait(_response_time)
            batch, prefetch_state = self._batch_queue.pop(0)
            self._not_full_cond.notify()
            return batch, prefetch_state

    # called from iterator
    def reset(self, prefetch_state):
        with self._lock:
            self._status = _Communicator.STATUS_RESET
            self._prefetch_state = prefetch_state
            self._batch_queue = []
            self._not_full_cond.notify()
            self._reset_count += 1

    # called from iterator
    def terminate(self):
        with self._lock:
            self._status = _Communicator.STATUS_TERMINATE
            self._batch_queue = []
            self._not_full_cond.notify()
            self._reset_count += 1

    # called from thread
    def check(self):
        with self._lock:
            status = self._status
            self._status = _Communicator.STATUS_CONTINUE
            prefetch_state = None
            if status == _Communicator.STATUS_RESET:
                prefetch_state = self._prefetch_state
            return status, prefetch_state, self._reset_count

    # called from thread
    def put(self, batch, prefetch_state, reset_count):
        with self._lock:
            if len(self._batch_queue) == self.n_prefetch:
                self._not_full_cond.wait()
            if reset_count == self._reset_count:
                self._batch_queue.append((batch, prefetch_state))
                self._not_empty_cond.notify()


class _PrefetchPipeline:
    _generate_id_thread = None
    _prefetch_batch_thread = None
    _remove_example_thread = None
    _terminating = False

    def __init__(self, dataset, batch_size, local_storage_base, n_generate_id, n_prefetch_from_backend,
                 n_prefetch_batch, n_remove_example, comm,
                 waiting_id_queue_max_size=0,
                 prefetched_id_queue_max_size=0,
                 used_id_queue_max_size=65536):
        self.dataset = dataset
        self.batch_size = batch_size
        self.local_storage_base = local_storage_base
        self.n_generate_id = n_generate_id
        self.n_prefetch_from_backend = n_prefetch_from_backend
        self.n_prefetch_batch = n_prefetch_batch
        self.n_remove_example = n_remove_example
        self._comm = comm

        self._waiting_id_queue = threading.Queue(waiting_id_queue_max_size)
        self._prefetched_id_queue = threading.Queue(prefetched_id_queue_max_size)
        self._used_id_queue = threading.Queue(used_id_queue_max_size)

    def launch_thread(self):
        pass
