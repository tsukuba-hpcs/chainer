import multiprocessing
import os
import queue
import shutil
import sys
import threading
import time
from multiprocessing import sharedctypes

import numpy

from chainer.dataset import iterator
from chainer.datasets.image_dataset import ExtendedLabeledImageDataset
from chainer.iterators._statemachine import IteratorState
from chainer.iterators._statemachine import iterator_statemachine
from chainer.iterators.order_samplers import ShuffleOrderSampler

_response_time = 0.1


def _solve_local_storage_path(local_storage_base, file_path):
    if file_path[0] == os.path.sep:
        return os.path.join(local_storage_base, file_path[1:])
    else:
        return os.path.join(local_storage_base, file_path)


class PrefetchMultiprocessIterator(iterator.Iterator):
    _finalized = False
    _comm = None
    _previous_epoch_detail = -1
    _prefetch_pipeline = None

    def __init__(self,
                 dataset,
                 batch_size,
                 local_storage_base,
                 n_prefetch,
                 n_prefetch_from_backend,
                 n_generate_batch,
                 n_remove_example=1,
                 repeat=True,
                 shuffle=None,
                 shared_mem=None,
                 dataset_timeout=30.0,
                 waiting_id_queue_max_size=1000,
                 prefetched_id_queue_max_size=1000,
                 used_id_queue_max_size=1000,
                 dataset_start=0,
                 dataset_finish=0,
                 measure=False
                 ):

        self.dataset = dataset  # support only ExtendedLabeledImageDataset
        self.batch_size = batch_size
        self.local_storage_base = local_storage_base
        self.repeat = repeat
        self.shuffle = shuffle
        self.n_prefetch = n_prefetch
        self.n_prefetch_from_backend = n_prefetch_from_backend
        self.n_generate_batch = n_generate_batch
        self.n_remove_example = n_remove_example
        self.order_sampler = ShuffleOrderSampler()  # fixed, for now
        self.shared_mem = shared_mem
        self.dataset_timeout = dataset_timeout
        self.dataset_start = dataset_start
        self.dataset_finish = dataset_finish
        self._measure = measure
        self.iteration = 0

        self._comm = _Communicator(self.n_prefetch, dataset_timeout)
        self.reset()

        self._prefetch_pipeline = _PrefetchPipeline(
            self.dataset,
            self.batch_size,
            self.local_storage_base,
            self.n_prefetch_from_backend,
            self.n_generate_batch,
            self.n_remove_example,
            self.shared_mem,
            self._comm,
            self.order_sampler,
            self.repeat,
            self.n_prefetch,
            waiting_id_queue_max_size,
            prefetched_id_queue_max_size,
            used_id_queue_max_size,
            dataset_start,
            dataset_finish,
            self._measure
        )

    def __next__(self):
        measure_mode = False
        if not self._prefetch_pipeline.launched:
            if self._prefetch_pipeline.measure_required():
                measure_mode = True
                batch, state = self._prefetch_pipeline.measure(
                    self.dataset_timeout)
            self._prefetch_pipeline.launch_thread()

        if not measure_mode:
            batch, state = self._comm.get()

        self._previous_epoch_detail = self.epoch_detail
        self._state = state

        if batch is None:
            raise StopIteration
        else:
            self.iteration += 1
            return batch

    def finalize(self):
        if self._finalized:
            return

        if self._comm is not None:
            self._comm.terminate()

        if self._prefetch_pipeline is not None and self._prefetch_pipeline.launched:
            self._prefetch_pipeline.terminate()

        self._comm = None
        self._prefetch_pipeline = None
        self._finalized = True

    @property
    def current_position(self):
        return self._state.current_position

    @property
    def epoch(self):
        return self._state.epoch

    @property
    def is_new_epoch(self):
        return self._state.is_new_epoch

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self._epoch_size

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    @property
    def _epoch_size(self):
        order = self._state.order
        if order is None:
            epoch_size = len(self.dataset)
        else:
            epoch_size = len(order)
        return epoch_size

    @property
    def task_time(self):
        return self._prefetch_pipeline.generate_batch_task_time

    @property
    def task_count(self):
        return self._prefetch_pipeline.generate_batch_task_count

    @property
    def cached_index_get_times(self):
        return self._prefetch_pipeline.cached_index_get_times

    @property
    def fetch_data_time(self):
        return self._prefetch_pipeline.fetch_data_time

    @property
    def unpack_and_organize_batch_time(self):
        return self._prefetch_pipeline.unpack_and_organize_batch_time

    @property
    def prefetch_time(self):
        return self._prefetch_pipeline.prefetch_time

    @property
    def generate_batch_times(self):
        return self._prefetch_pipeline.generate_batch_times

    @property
    def get_example_times(self):
        return self._prefetch_pipeline.get_example_times

    @property
    def read_data_times(self):
        return self._prefetch_pipeline.read_data_times

    @property
    def start_prefetch_process_times(self):
        return self._prefetch_pipeline.start_prefetch_process_times

    @property
    def generate_batch_pool_time(self):
        return self._prefetch_pipeline.generate_batch_pool_time

    @property
    def generate_batch_thread_and_start_time(self):
        return self._prefetch_pipeline.generate_batch_thread_and_start_time

    @property
    def launch_thread_time(self):
        return self._prefetch_pipeline.launch_thread_time

    @property
    def start_generate_random_id_process_time(self):
        return self._prefetch_pipeline.start_generate_random_id_process_time

    @property
    def initial_prefetch_multiprocess_iterator_cached_id_queue_size(self):
        return self._prefetch_pipeline.initial_prefetch_multiprocess_iterator_cached_id_queue_size

    def reset(self):
        order = self.order_sampler(numpy.arange(len(self.dataset)), 0)
        self._reset_state(0, 0, False, order)
        self._previous_epoch_detail = -1

    def _reset_state(self, current_position, epoch, is_new_epoch, order):
        self._state = IteratorState(
            current_position, epoch, is_new_epoch, order)
        self._comm.reset(self._state)


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
        if self.is_terminated:
            return

        with self._lock:
            if len(self._batch_queue) == self.n_prefetch:
                self._not_full_cond.wait()
            if reset_count == self._reset_count:
                self._batch_queue.append((batch, prefetch_state))
                self._not_empty_cond.notify()


_prefetch_multiprocess_iterator_fetch_dataset = None
_prefetch_multiprocess_iterator_mem_size = None
_prefetch_multiprocess_iterator_mem_bulk = None
_prefetch_multiprocess_iterator_local_storage_base = None
_prefetch_multiprocess_iterator_terminating = None

_prefetch_multiprocess_iterator_waiting_id_queue = None
_prefetch_multiprocess_iterator_cached_id_queue = None
_prefetch_multiprocess_iterator_used_id_queue = None


class _PrefetchPipeline:
    _generate_random_id_process = None
    _prefetch_from_backend_thread = None
    _generate_batch_thread = None
    _remove_example_thread = None
    _generate_prefetch_process_thread = None
    _remove_example_process = None

    _prefetch_from_backend_pool = []
    _generate_batch_pool = None
    _remove_example_pool = None
    _terminating = False
    _launched = False

    def __init__(self,
                 dataset: ExtendedLabeledImageDataset,
                 batch_size,
                 local_storage_base,
                 n_prefetch_from_backend,
                 n_generate_batch,
                 n_remove_example,
                 mem_size,
                 comm,
                 order_sampler,
                 repeat,
                 prefetch_batch_size,
                 waiting_id_queue_max_size,
                 prefetched_id_queue_max_size,
                 used_id_queue_max_size,
                 dataset_start=0,
                 dataset_finish=0,
                 measure=False
                 ):

        self.dataset = dataset
        # cannot pickle the thread.lock object included in dataset object
        self.dataset_pairs = self.dataset.pairs
        self.dataset_root = self.dataset.root

        self.batch_size = batch_size
        self.local_storage_base = local_storage_base
        self.n_prefetch_from_backend = n_prefetch_from_backend
        self.n_generate_batch = n_generate_batch
        self.n_remove_example = n_remove_example
        self.mem_size = mem_size
        self._comm = comm
        self.order_sampler = order_sampler
        self.repeat = repeat
        self.prefetch_batch_size = prefetch_batch_size
        self._measure = measure

        self.dataset_start = dataset_start
        self.dataset_finish = dataset_finish
        self._generate_batch_task_time = 0
        self._generate_batch_task_count = 0
        self._cached_index_get_times = []
        self._fetch_data_time = 0
        self._unpack_and_organize_batch_time = 0
        self._prefetch_time = {}
        self.count = 0
        self._generate_batch_times = []
        self._get_example_times = []
        self._read_data_times = []
        self._start_prefetch_process_times = []
        self._generate_batch_pool_time = 0
        self._generate_batch_thread_and_start_time = 0
        self._launch_thread_time = 0
        self._start_generate_random_id_process_time = 0
        self._initial_prefetch_multiprocess_iterator_cached_id_queue_size = 0

        self._allocate_shared_memory()

        global _prefetch_multiprocess_iterator_waiting_id_queue
        global _prefetch_multiprocess_iterator_cached_id_queue
        # global _prefetch_multiprocess_iterator_used_id_queue
        _prefetch_multiprocess_iterator_waiting_id_queue = multiprocessing.Queue(waiting_id_queue_max_size)
        _prefetch_multiprocess_iterator_waiting_id_queue.cancel_join_thread()
        _prefetch_multiprocess_iterator_cached_id_queue = multiprocessing.Queue(prefetched_id_queue_max_size)
        _prefetch_multiprocess_iterator_cached_id_queue.cancel_join_thread()
        # _prefetch_multiprocess_iterator_used_id_queue = multiprocessing.Queue(used_id_queue_max_size)
        # _prefetch_multiprocess_iterator_used_id_queue.cancel_join_thread()

        global _prefetch_multiprocess_iterator_terminating
        _prefetch_multiprocess_iterator_terminating = multiprocessing.Event()

    @property
    def launched(self):
        return self._launched

    @property
    def generate_batch_task_time(self):
        return self._generate_batch_task_time

    @generate_batch_task_time.setter
    def generate_batch_task_time(self, task_time):
        self._generate_batch_task_time = task_time

    @property
    def generate_batch_task_count(self):
        return self._generate_batch_task_count

    @generate_batch_task_count.setter
    def generate_batch_task_count(self, task_count):
        self._generate_batch_task_count = task_count

    @property
    def cached_index_get_times(self):
        return self._cached_index_get_times

    @property
    def fetch_data_time(self):
        return self._fetch_data_time

    @fetch_data_time.setter
    def fetch_data_time(self, fetch_data_time):
        self._fetch_data_time = fetch_data_time

    @property
    def unpack_and_organize_batch_time(self):
        return self._unpack_and_organize_batch_time

    @unpack_and_organize_batch_time.setter
    def unpack_and_organize_batch_time(self, unpack_and_organize_batch_time):
        self._unpack_and_organize_batch_time = unpack_and_organize_batch_time

    @property
    def prefetch_time(self):
        return self._prefetch_time

    @property
    def generate_batch_times(self):
        return self._generate_batch_times

    @property
    def get_example_times(self):
        return self._get_example_times

    @property
    def read_data_times(self):
        return self._read_data_times

    @property
    def start_prefetch_process_times(self):
        return self._start_prefetch_process_times

    @property
    def generate_batch_pool_time(self):
        return self._generate_batch_pool_time

    @property
    def generate_batch_thread_and_start_time(self):
        return self._generate_batch_thread_and_start_time

    @property
    def launch_thread_time(self):
        return self._launch_thread_time

    @property
    def start_generate_random_id_process_time(self):
        return self._start_generate_random_id_process_time

    @property
    def initial_prefetch_multiprocess_iterator_cached_id_queue_size(self):
        return self._initial_prefetch_multiprocess_iterator_cached_id_queue_size

    def reset_all_timers(self):
        self._generate_batch_task_time = 0
        self._cached_index_get_times = []
        self._fetch_data_time = 0
        self._unpack_and_organize_batch_time = 0
        self._prefetch_time = {}
        self._generate_batch_times = {}
        self._get_example_times = {}
        self._read_data_times = {}
        self._start_prefetch_process_times = []
        self._generate_batch_pool_time = 0
        self._generate_batch_thread_and_start_time = 0
        self._launch_thread_time = 0
        self._start_generate_random_id_process_time = 0

    def reset_all_counts(self):
        self._generate_batch_task_count = 0
        self._initial_prefetch_multiprocess_iterator_cached_id_queue_size = 0

    def measure_required(self):
        return self.mem_size is None

    def measure(self, dataset_timeout):
        # dataset_timeout: timeout in seconds or None

        status, prefetch_state, _ = self._comm.check()
        if status == _Communicator.STATUS_RESET:
            self.prefetch_state = prefetch_state

        self.prefetch_state, indices = iterator_statemachine(
            self.prefetch_state, self.batch_size, self.repeat,
            self.order_sampler, len(self.dataset))
        if indices is None:  # stop iteration
            batch = None
        else:
            batch_ret = [None]

            def fetch_batch():
                batch_ret[0] = [self.dataset[idx] for idx in indices]

            if dataset_timeout is None:
                # Timeout is not set: fetch synchronously
                fetch_batch()
            else:
                # Timeout is set: fetch asynchronously and watch for timeout
                thr = threading.Thread(target=fetch_batch)
                thr.daemon = True
                thr.start()
                thr.join(dataset_timeout)
                if thr.is_alive():
                    _raise_timeout_warning()
                thr.join()

            batch = batch_ret[0]
            self.mem_size = max(map(_measure, batch))
            self._allocate_shared_memory()

        return batch, self.prefetch_state

    def _allocate_shared_memory(self):
        if self.measure_required():
            self.mem_bulk = None
        else:
            self.mem_bulk = \
                sharedctypes.RawArray('b', self.batch_size * self.mem_size)

    def launch_thread(self):
        launch_thread_timer = time.time()  # timer

        global _prefetch_multiprocess_iterator_fetch_dataset
        global _prefetch_multiprocess_iterator_local_storage_base
        _prefetch_multiprocess_iterator_fetch_dataset = self.dataset
        _prefetch_multiprocess_iterator_local_storage_base = self.local_storage_base

        start_generate_random_id_process_timer = time.time()  # timer
        self._generate_random_id_process = multiprocessing.Process(
            target=_generate_random_id_loop,
            args=[
                _prefetch_multiprocess_iterator_terminating,
                _prefetch_multiprocess_iterator_waiting_id_queue,
                _prefetch_multiprocess_iterator_fetch_dataset,
                self.dataset_start,
                self.prefetch_batch_size,
                self.batch_size,
                self.repeat,
                self.order_sampler
            ],
            daemon=True
        )
        self._generate_random_id_process.start()
        self._start_generate_random_id_process_time = time.time() - start_generate_random_id_process_timer  # timer

        self._generate_prefetch_process_thread = threading.Thread(
            target=self._prefetch_from_backend_loop,
            name='_generate_prefetch_process_thread'
        )
        self._generate_prefetch_process_thread.daemon = True
        self._generate_prefetch_process_thread.start()

        generate_batch_pool_timer = time.time()  # timer
        self._generate_batch_pool = multiprocessing.Pool(
            processes=self.n_generate_batch,
            initializer=_fetch_setup,
            initargs=(
                self.mem_size,
                self.mem_bulk,
                _prefetch_multiprocess_iterator_fetch_dataset,
                _prefetch_multiprocess_iterator_local_storage_base
            )
        )
        self._generate_batch_pool_time = time.time() - generate_batch_pool_timer  # timer
        generate_batch_thread_and_start_timer = time.time()  # timer
        self._generate_batch_thread = threading.Thread(
            target=self._generate_batch_loop,
            name='_generate_batch_loop'
        )
        self._generate_batch_thread.daemon = True
        self._generate_batch_thread.start()
        self._generate_batch_thread_and_start_time = time.time() - generate_batch_thread_and_start_timer  # timer

        '''
        self._remove_example_process = multiprocessing.Process(
            target=_remove_example_loop,
            args=[
                _prefetch_multiprocess_iterator_terminating,
                _prefetch_multiprocess_iterator_used_id_queue,
                _prefetch_multiprocess_iterator_fetch_dataset,
                _prefetch_multiprocess_iterator_local_storage_base
            ],
            daemon=True
        )
        self._remove_example_process.start()
        '''

        self._launched = True
        self._launch_thread_time = time.time() - launch_thread_timer  # timer

    def terminate(self):
        _prefetch_multiprocess_iterator_terminating.set()

        if self._generate_random_id_process is not None:
            self._generate_random_id_process.terminate()
            self._generate_random_id_process.join()
            self._generate_random_id_process = None

        if self._prefetch_from_backend_pool is not None:
            for process in self._prefetch_from_backend_pool:
                process.terminate()
                process.join()
            self._prefetch_from_backend_pool = None
            self._generate_prefetch_process_thread.join()

        if self._remove_example_process is not None:
            self._remove_example_process.terminate()
            self._remove_example_process = None

        '''
        for thread in [self._generate_batch_thread]:
            if thread is not None:
                while thread.is_alive():
                    thread.join(_response_time)
                thread = None
        '''

        if self._generate_batch_pool is not None:
            self._generate_batch_pool.close()
            self._generate_batch_pool.terminate()
            self._generate_batch_pool.join()

            self._generate_batch_pool = None

        self._launched = False

    def _prefetch_from_backend_loop(self):
        self._prefetch_from_backend_pool = []
        for _ in range(self.n_prefetch_from_backend):
            start_process_timer = time.time()  # timer
            process = multiprocessing.Process(
                target=_prefetch_from_backend,
                args=[
                    _prefetch_multiprocess_iterator_terminating,
                    _prefetch_multiprocess_iterator_waiting_id_queue,
                    _prefetch_multiprocess_iterator_cached_id_queue,
                    _prefetch_multiprocess_iterator_fetch_dataset,
                    _prefetch_multiprocess_iterator_local_storage_base
                ]
            )
            process.start()
            self._prefetch_from_backend_pool.append(process)
            start_process_time = time.time() - start_process_timer  # timer
            self._start_prefetch_process_times.append(start_process_time)  # timer

    def _generate_batch_loop(self):
        alive = True
        try:
            self._initial_prefetch_multiprocess_iterator_cached_id_queue_size = \
                _prefetch_multiprocess_iterator_cached_id_queue.qsize()
        except NotImplementedError:
            print('WARNING:  _prefetch_multiprocess_iterator_cached_id_queue.qsize() raise NotImplementedError',
                  file=sys.stderr)

        while alive:
            if _prefetch_multiprocess_iterator_terminating.is_set():
                break
            task_timer = time.time()
            alive = self._generate_batch_task()
            self.generate_batch_task_time = self.generate_batch_task_time + time.time() - task_timer
            self.generate_batch_task_count = self.generate_batch_task_count + 1

    def _generate_batch_task(self):
        status, prefetch_state, reset_count = self._comm.check()

        if status == _Communicator.STATUS_RESET:
            self.prefetch_state = prefetch_state
        elif status == _Communicator.STATUS_TERMINATE:
            return False  # stop loop

        # Here, indices is used only to decide whether iteration should be stopped or not
        self.prefetch_state, _indices = iterator_statemachine(
            self.prefetch_state, self.batch_size, self.repeat,
            self.order_sampler, len(self.dataset))
        # if repeat == False and passed 1 epoch, `indices` will be None
        # see the implementation of `iterator_statemachine` for more detail
        if _indices is None:
            batch = None
        else:
            cached_index_get_timer = time.time()
            while True:
                try:
                    indices, prefetcher_pid, prefetch_time \
                        = _prefetch_multiprocess_iterator_cached_id_queue.get(timeout=_response_time)
                except queue.Empty:
                    if _prefetch_multiprocess_iterator_terminating.is_set():
                        return False
                else:
                    break
            cached_index_get_time = time.time() - cached_index_get_timer
            self.cached_index_get_times.append(cached_index_get_time)
            fetch_data_timer = time.time()
            future = self._generate_batch_pool.map_async(_generate_batch, enumerate(indices))
            while True:
                try:
                    data_all = future.get(_response_time)
                except multiprocessing.TimeoutError:
                    if _prefetch_multiprocess_iterator_terminating.is_set():
                        return False
                else:
                    break
            self.fetch_data_time = self.fetch_data_time + time.time() - fetch_data_timer

            unpack_and_organize_batch_timer = time.time()
            batch = [_unpack(data[0], self.mem_bulk) for data in data_all]
            self.unpack_and_organize_batch_time = self.unpack_and_organize_batch_time + time.time() - \
                                                  unpack_and_organize_batch_timer

            if self._measure:
                for data in data_all:
                    self._generate_batch_times.append(data[1])
                    self._read_data_times.append(data[0][2])
                    self._get_example_times.append(data[0][3])

                if prefetcher_pid not in self._prefetch_time.keys():
                    self._prefetch_time[prefetcher_pid] = []
                self._prefetch_time[prefetcher_pid].append(prefetch_time)

        self._comm.put(batch, self.prefetch_state, reset_count)

        '''
        if batch is not None:
            while True:
                try:
                    _prefetch_multiprocess_iterator_used_id_queue.put(indices, timeout=_response_time)
                except queue.Full:
                    if _prefetch_multiprocess_iterator_terminating.is_set():
                        return False
                else:
                    break
        '''

        return True


# _PrefetchPipeline object includes _Communicator object, which includes thread.lock()
# So _PrefetchPipeline object is "unpicklabele", and cannot be used in a multiprocessing.Process function.
# This is because why some methods which are called by multiprocessing.Process are defined as static method.

def _fetch_setup(mem_size, mem_bulk, dataset, local_storage_base):
    global _prefetch_multiprocess_iterator_mem_size
    global _prefetch_multiprocess_iterator_mem_bulk
    global _prefetch_multiprocess_iterator_fetch_dataset
    global _prefetch_multiprocess_iterator_local_storage_base
    _prefetch_multiprocess_iterator_mem_size = mem_size
    _prefetch_multiprocess_iterator_mem_bulk = mem_bulk
    _prefetch_multiprocess_iterator_fetch_dataset = dataset
    _prefetch_multiprocess_iterator_local_storage_base = local_storage_base


def _generate_random_id_loop(
        _prefetch_multiprocess_iterator_terminating,
        _prefetch_multiprocess_iterator_waiting_id_queue,
        _prefetch_multiprocess_iterator_fetch_dataset,
        dataset_start,
        prefetch_batch_size,
        batch_size,
        repeat,
        order_sampler
):
    _prefetch_multiprocess_iterator_waiting_id_queue.cancel_join_thread()
    dataset_length = len(_prefetch_multiprocess_iterator_fetch_dataset)
    initial_order = order_sampler(numpy.arange(dataset_length), 0)
    random_id_state = IteratorState(0, 0, False, initial_order)

    '''
    print(f'{os.uname()[1]}/{os.getpid()}:dataset_length:{dataset_length}', file=sys.stderr)
    print(f'{os.uname()[1]}/{os.getpid()}:len(initial_order):{len(initial_order)}', file=sys.stderr)
    print(f'{os.uname()[1]}/{os.getpid()}:random_id_state:{random_id_state}', file=sys.stderr)
    print(f'{os.uname()[1]}/{os.getpid()}:len(numpy.unique(initial_order)):{len(numpy.unique(initial_order))}', file=sys.stderr)
    print(f'{os.uname()[1]}/{os.getpid()}:dataset_start:{dataset_start}', file=sys.stderr)
    sys.stderr.flush()
    '''

    while not _prefetch_multiprocess_iterator_terminating.is_set():
        random_id_state, indices = iterator_statemachine(
            random_id_state,
            batch_size,
            repeat,
            order_sampler,
            dataset_length
        )

        while True:
            try:
                # Note: `indices` is an object of numpy.ndarray
                _prefetch_multiprocess_iterator_waiting_id_queue.put(dataset_start + indices, timeout=_response_time)
            except queue.Full:
                if _prefetch_multiprocess_iterator_terminating.is_set():
                    return
            else:
                break


def _prefetch_from_backend(
        _prefetch_multiprocess_iterator_terminating,
        _prefetch_multiprocess_iterator_waiting_id_queue,
        _prefetch_multiprocess_iterator_cached_id_queue,
        _prefetch_multiprocess_iterator_fetch_dataset,
        _prefetch_multiprocess_iterator_local_storage_base
):
    _prefetch_multiprocess_iterator_waiting_id_queue.cancel_join_thread()
    _prefetch_multiprocess_iterator_cached_id_queue.cancel_join_thread()

    my_pid = os.getpid()
    while True:
        prefetch_from_backend_timer = time.time()
        while not _prefetch_multiprocess_iterator_terminating.is_set():
            try:
                indices = _prefetch_multiprocess_iterator_waiting_id_queue.get(timeout=_response_time)
            except queue.Empty:
                if _prefetch_multiprocess_iterator_terminating.is_set():
                    return False
            else:
                break

        cache_hit_count = 0
        for index in indices:
            backend_storage_file_path = os.path.join(_prefetch_multiprocess_iterator_fetch_dataset.root,
                                                     _prefetch_multiprocess_iterator_fetch_dataset.pairs[index][0])
            local_storage_file_path = _solve_local_storage_path(_prefetch_multiprocess_iterator_local_storage_base,
                                                                backend_storage_file_path)
            if not os.path.exists(local_storage_file_path):
                os.makedirs(os.sep.join(local_storage_file_path.split(os.sep)[:-1]), exist_ok=True)
                shutil.copyfile(backend_storage_file_path, f'{local_storage_file_path}.{my_pid}')
                os.rename(f'{local_storage_file_path}.{my_pid}', local_storage_file_path)
            else:
                cache_hit_count += 1
        prefetch_from_backend_time = time.time() - prefetch_from_backend_timer

        while True:
            try:
                _prefetch_multiprocess_iterator_cached_id_queue.put([indices, my_pid, prefetch_from_backend_time],
                                                                    timeout=_response_time)
            except queue.Full:
                if _prefetch_multiprocess_iterator_terminating.is_set():
                    return False
            else:
                break


def _generate_batch(inputs):
    generate_batch_timer = time.time()
    i, index = inputs
    path, int_label = _prefetch_multiprocess_iterator_fetch_dataset.pairs[index]
    backend_storage_file_path = os.path.join(_prefetch_multiprocess_iterator_fetch_dataset.root, path)
    local_storage_file_path = _solve_local_storage_path(_prefetch_multiprocess_iterator_local_storage_base,
                                                        backend_storage_file_path)

    if os.path.exists(local_storage_file_path):
        data = _prefetch_multiprocess_iterator_fetch_dataset.get_example_by_path(
            local_storage_file_path,
            int_label
        )
    else:
        data = _prefetch_multiprocess_iterator_fetch_dataset.get_example_by_path(
            path,
            int_label
        )

    if _prefetch_multiprocess_iterator_mem_bulk is not None:
        offset = i * _prefetch_multiprocess_iterator_mem_size
        limit = offset + _prefetch_multiprocess_iterator_mem_size
        data = _pack(data, _prefetch_multiprocess_iterator_mem_bulk, offset, limit)

    generate_batch_time = time.time() - generate_batch_timer
    return data, generate_batch_time


def _remove_example_loop(
        _prefetch_multiprocess_iterator_terminating,
        _prefetch_multiprocess_iterator_used_id_queue,
        _prefetch_multiprocess_iterator_fetch_dataset,
        _prefetch_multiprocess_iterator_local_storage_base
):
    _prefetch_multiprocess_iterator_used_id_queue.cancel_join_thread()

    while not _prefetch_multiprocess_iterator_terminating.is_set():
        try:
            if _prefetch_multiprocess_iterator_used_id_queue.full():
                indeces = _prefetch_multiprocess_iterator_used_id_queue.get(timeout=_response_time)
                for index in indeces:
                    _remove_example(
                        _prefetch_multiprocess_iterator_terminating,
                        _prefetch_multiprocess_iterator_fetch_dataset,
                        _prefetch_multiprocess_iterator_local_storage_base,
                        index
                    )
            else:
                continue
        except queue.Empty:
            if _prefetch_multiprocess_iterator_terminating.is_set():
                return False


def _remove_example(
        _prefetch_multiprocess_iterator_terminating,
        _prefetch_multiprocess_iterator_fetch_dataset,
        _prefetch_multiprocess_iterator_local_storage_base,
        index
):
    if not _prefetch_multiprocess_iterator_terminating.is_set():
        path, _ = _prefetch_multiprocess_iterator_fetch_dataset.pairs[index]

        backend_storage_file_path = os.path.join(_prefetch_multiprocess_iterator_fetch_dataset.root, path)
        delete_file_path = _solve_local_storage_path(_prefetch_multiprocess_iterator_local_storage_base,
                                                     backend_storage_file_path)
        if os.path.exists(delete_file_path):
            try:
                os.remove(delete_file_path)
            except FileNotFoundError:
                pass


# copied from multiprocess_iterator.py
class _PackedNdarray(object):

    def __init__(self, array, mem, offset):
        self.shape = array.shape
        self.dtype = array.dtype
        self.nbytes = array.nbytes
        self.size = array.size
        self.offset = offset
        total = self.offset + self.nbytes
        if total > len(mem):
            raise ValueError(
                'Shared memory size is too small. expect:{}, actual:{}'.format(
                    total, len(mem)))
        target = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        target[...] = array.ravel()

    def unpack(self, mem):
        ret = numpy.frombuffer(mem, self.dtype, self.size, self.offset)
        ret = ret.reshape(self.shape).copy()
        return ret


def _measure(data):
    expect = 0
    t = type(data)
    if t is tuple or t is list or t is dict:
        for v in data:
            if isinstance(v, numpy.ndarray):
                expect += v.nbytes
    return expect


def _pack(data, mem, offset, limit):
    if len(mem) == 0:
        return data
    t = type(data)
    over = False
    if t is tuple or t is list:
        ret = []
        for v in data:
            if isinstance(v, numpy.ndarray):
                if v.nbytes + offset > limit:
                    over = True
                else:
                    v = _PackedNdarray(v, mem, offset)
                    offset += v.nbytes
            ret.append(v)
        data = t(ret)
    elif t is dict:
        ret = {}
        for k, v in six.iteritems(data):
            if isinstance(v, numpy.ndarray):
                if v.nbytes + offset > limit:
                    over = True
                else:
                    v = _PackedNdarray(v, mem, offset)
                    offset += v.nbytes
            ret[k] = v
        data = ret
    elif t is numpy.ndarray:
        if data.nbytes + offset > limit:
            over = True
        else:
            data = _PackedNdarray(data, mem, offset)
            offset += data.nbytes
    if over:
        expect = _measure(data)
        warnings.warn(
            'Shared memory size is too small.\n' +
            'Please set shared_mem option for MultiprocessIterator.\n' +
            'Expect shared memory size: {} bytes.\n'.format(expect) +
            'Actual shared memory size: {} bytes.'.format(limit - offset),
            UserWarning)
    return data


def _unpack(data, mem):
    if len(mem) == 0:
        return data
    t = type(data)
    if t is tuple or t is list:
        ret = []
        for v in data:
            if isinstance(v, _PackedNdarray):
                v = v.unpack(mem)
            ret.append(v)
        data = t(ret)
    elif t is dict:
        ret = {}
        for k, v in six.iteritems(data):
            if isinstance(v, _PackedNdarray):
                v = v.unpack(mem)
            ret[k] = v
        data = ret
    elif t is _PackedNdarray:
        data = data.unpack(mem)
    return data
