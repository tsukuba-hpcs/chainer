import multiprocessing
import os
import queue
import shutil
import threading

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

    def __init__(self, dataset: ExtendedLabeledImageDataset,
                 batch_size, local_storage_base,
                 n_prefetch,
                 n_prefetch_from_backend, n_generate_batch,
                 n_remove_example, repeat=True, shuffle=None, dataset_timeout=30.0,
                 waiting_id_queue_max_size=1000,
                 prefetched_id_queue_max_size=1000,
                 used_id_queue_max_size=1000
                 ):
        if type(dataset) is not ExtendedLabeledImageDataset:
            raise AssertionError('This iterator only supports `ExtendedLabeledImageDataset`')

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

        self._comm = _Communicator(self.n_prefetch, dataset_timeout)
        self.reset()

        self._prefetch_pipeline = _PrefetchPipeline(
            self.dataset, self.batch_size, self.local_storage_base,
            self.n_prefetch_from_backend, self.n_generate_batch,
            self.n_remove_example, self._comm, self.order_sampler,
            self.repeat,
            self.n_prefetch,
            waiting_id_queue_max_size,
            prefetched_id_queue_max_size,
            used_id_queue_max_size
        )

    def __next__(self):
        if not self._prefetch_pipeline.launched:
            self._prefetch_pipeline.launch_thread()

        batch, state = self._comm.get()
        self._previous_epoch_detail = self.epoch_detail
        self._state = state

        if batch is None:
            raise StopIteration
        else:
            return batch

    def finalize(self):
        if self._finalized:
            return

        if self._comm is not None:
            self._comm.terminate()

        if self._prefetch_pipeline.launched:
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
        if self._status == _Communicator.STATUS_TERMINATE:
            return

        with self._lock:
            if len(self._batch_queue) == self.n_prefetch:
                self._not_full_cond.wait()
            if reset_count == self._reset_count:
                self._batch_queue.append((batch, prefetch_state))
                self._not_empty_cond.notify()


_prefetch_multiprocess_iterator_fetch_dataset = None
_prefetch_multiprocess_iterator_local_storage_base = None
_prefetch_multiprocess_iterator_terminating = False

_prefetch_multiprocess_iterator_waiting_id_queue = None
_prefetch_multiprocess_iterator_cached_id_queue = None
_prefetch_multiprocess_iterator_used_id_queue = None


class _PrefetchPipeline:
    _generate_random_id_thread = None
    _prefetch_from_backend_thread = None
    _generate_batch_thread = None
    _remove_example_thread = None

    _prefetch_from_backend_pool = None
    _generate_batch_pool = None
    _remove_example_pool = None
    _terminating = False
    _launched = False

    def __init__(self, dataset: ExtendedLabeledImageDataset,
                 batch_size, local_storage_base,
                 n_prefetch_from_backend,
                 n_generate_batch, n_remove_example, comm, order_sampler,
                 repeat,
                 prefetch_batch_size,
                 waiting_id_queue_max_size,
                 prefetched_id_queue_max_size,
                 used_id_queue_max_size):
        if type(dataset) is not ExtendedLabeledImageDataset:
            raise AssertionError('This iterator only supports `ExtendedLabeledImageDataset`')

        self.dataset = dataset
        # cannot pickle the thread.lock object included in dataset object
        self.dataset_pairs = self.dataset.pairs
        self.dataset_root = self.dataset.root

        self.batch_size = batch_size
        self.local_storage_base = local_storage_base
        self.n_prefetch_from_backend = n_prefetch_from_backend
        self.n_generate_batch = n_generate_batch
        self.n_remove_example = n_remove_example
        self._comm = comm
        self.order_sampler = order_sampler
        self.repeat = repeat
        self.prefetch_batch_size = prefetch_batch_size

        initial_order = self.order_sampler(numpy.arange(len(self.dataset)), 0)
        self._random_id_state = IteratorState(0, 0, False, initial_order)

        global _prefetch_multiprocess_iterator_waiting_id_queue
        global _prefetch_multiprocess_iterator_cached_id_queue
        global _prefetch_multiprocess_iterator_used_id_queue
        _prefetch_multiprocess_iterator_waiting_id_queue = queue.Queue(waiting_id_queue_max_size)
        _prefetch_multiprocess_iterator_cached_id_queue = queue.Queue(prefetched_id_queue_max_size)
        _prefetch_multiprocess_iterator_used_id_queue = queue.Queue(used_id_queue_max_size)

    @property
    def launched(self):
        return self._launched

    def launch_thread(self):
        self._fetch_setup(self.dataset, self.local_storage_base)

        self._generate_random_id_thread = threading.Thread(
            target=self._generate_random_id_loop,
            name='_generate_random_id_loop'
        )
        self._generate_random_id_thread.start()

        self._prefetch_from_backend_pool = multiprocessing.Pool(processes=self.n_prefetch_from_backend)
        self._prefetch_from_backend_thread = threading.Thread(
            target=self._prefetch_from_backend_loop,
            name='_prefetch_from_backend_loop'
        )
        self._prefetch_from_backend_thread.daemon = True
        self._prefetch_from_backend_thread.start()

        self._generate_batch_pool = multiprocessing.Pool(processes=self.n_generate_batch)
        self._generate_batch_thread = threading.Thread(
            target=self._generate_batch_loop,
            name='_generate_batch_loop'
        )
        self._generate_batch_thread.daemon = True
        self._generate_batch_thread.start()

        self._remove_example_pool = multiprocessing.Pool(processes=self.n_remove_example)
        self._remove_example_thread = threading.Thread(
            target=self._remove_example_loop,
            name='_remove_example_loop'
        )
        self._remove_example_thread.daemon = True
        self._remove_example_thread.start()

        self._launched = True

    def terminate(self):
        global _prefetch_multiprocess_iterator_terminating
        _prefetch_multiprocess_iterator_terminating = True
        for thread in [self._remove_example_thread,
                       self._prefetch_from_backend_thread,
                       self._generate_random_id_thread,
                       self._generate_batch_thread]:
            if thread is not None:
                while thread.is_alive():
                    thread.join(_response_time)

        self._prefetch_from_backend_pool.terminate()
        self._generate_batch_pool.terminate()

        self._launched = False

    def _fetch_setup(self, dataset, local_storage_base):
        global _pfetch_multiprocess_iterator_fetch_dataset
        global _prefetch_multiprocess_iterator_local_storage_base
        _pfetch_multiprocess_iterator_fetch_dataset = dataset
        _prefetch_multiprocess_iterator_local_storage_base = local_storage_base

    def _generate_random_id_loop(self):
        indices_list = []
        while not _prefetch_multiprocess_iterator_terminating:
            if not indices_list:
                for _ in range(self.prefetch_batch_size):
                    self._random_id_state, indices = iterator_statemachine(
                        self._random_id_state,
                        self.batch_size,
                        self.repeat,
                        self.order_sampler,
                        len(self.dataset)
                    )
                    indices_list.append(indices)
            try:
                _prefetch_multiprocess_iterator_waiting_id_queue.put(numpy.array(indices_list),
                                                                     timeout=_response_time)
                indices_list = []
            except queue.Full:
                if _prefetch_multiprocess_iterator_terminating:
                    break

    def _prefetch_from_backend_loop(self):
        alive = True
        try:
            while alive:
                if _prefetch_multiprocess_iterator_terminating:
                    break
                alive = self._prefetch_from_backend_task()
        finally:
            self._prefetch_from_backend_pool.close()
            self._prefetch_from_backend_pool.join()

    def _prefetch_from_backend_task(self):
        try:
            indices_list = _prefetch_multiprocess_iterator_waiting_id_queue.get(timeout=_response_time)
            future = self._prefetch_from_backend_pool.map_async(_prefetch_from_backend, indices_list)
            _ = future.get(timeout=_response_time)

            for indices in indices_list:
                _prefetch_multiprocess_iterator_cached_id_queue.put(indices, timeout=_response_time)
        except queue.Empty:
            if _prefetch_multiprocess_iterator_terminating:
                return False
        except multiprocessing.TimeoutError:
            if _prefetch_multiprocess_iterator_terminating:
                return False
        except queue.Full:
            if _prefetch_multiprocess_iterator_terminating:
                return False

        return True

    def _generate_batch_loop(self):
        alive = True
        try:
            while alive:
                if _prefetch_multiprocess_iterator_terminating:
                    break
                alive = self._generate_batch_task()
        finally:
            self._generate_batch_pool.close()
            self._generate_batch_pool.join()

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
            indices = []
            while True:
                try:
                    indices = _prefetch_multiprocess_iterator_cached_id_queue.get(timeout=_response_time)
                except queue.Empty:
                    if _prefetch_multiprocess_iterator_terminating:
                        return False
                else:
                    break

            future = self._generate_batch_pool.map_async(_generate_batch, indices)
            while True:
                try:
                    batch = future.get(_response_time)
                except multiprocessing.TimeoutError:
                    if _prefetch_multiprocess_iterator_terminating:
                        return False
                else:
                    break

            try:
                _prefetch_multiprocess_iterator_used_id_queue.put(indices, timeout=_response_time)
            except queue.Full:
                if _prefetch_multiprocess_iterator_terminating:
                    return False

        self._comm.put(batch, self.prefetch_state, reset_count)
        return True

    def _remove_example_loop(self):
        alive = True
        try:
            while alive:
                if _prefetch_multiprocess_iterator_terminating:
                    break
                alive = self._remove_example_task()
        finally:
            self._remove_example_pool.close()
            # Somehow, only `self_remove_example_pool` must be terminated,
            # or this thread cannot be joined.
            self._remove_example_pool.terminate()
            self._remove_example_pool.join()

    def _remove_example_task(self):
        try:
            if _prefetch_multiprocess_iterator_used_id_queue.full():
                index = _prefetch_multiprocess_iterator_used_id_queue.get(timeout=_response_time)
                self._remove_example_pool.apply_async(_remove_example, args=[index])
        except queue.Empty:
            if _prefetch_multiprocess_iterator_terminating:
                return False

        return True


# _PrefetchPipeline object includes _Communicator object, which includes thread.lock()
# So _PrefetchPipeline object is "unpicklabele", and cannot be used in a multiprocessing.Process function.
# This is because why some methods which are called by multiprocessing.Process are defined as static method.

def _prefetch_from_backend(indices):
    for index in indices:
        backend_storage_file_path = os.path.join(_pfetch_multiprocess_iterator_fetch_dataset.root,
                                                 _pfetch_multiprocess_iterator_fetch_dataset.pairs[index][0])
        local_storage_file_path = _solve_local_storage_path(_prefetch_multiprocess_iterator_local_storage_base,
                                                            backend_storage_file_path)
        my_pid = os.getpid()
        if not os.path.exists(local_storage_file_path):
            os.makedirs(os.sep.join(local_storage_file_path.split(os.sep)[:-1]), exist_ok=True)
            shutil.copyfile(backend_storage_file_path, f'{local_storage_file_path}.{my_pid}')
            os.rename(f'{local_storage_file_path}.{my_pid}', local_storage_file_path)


def _generate_batch(index):
    path, int_label = _pfetch_multiprocess_iterator_fetch_dataset.pairs[index]
    backend_storage_file_path = os.path.join(_pfetch_multiprocess_iterator_fetch_dataset.root, path)
    local_storage_file_path = _solve_local_storage_path(_prefetch_multiprocess_iterator_local_storage_base,
                                                        backend_storage_file_path)
    data = _pfetch_multiprocess_iterator_fetch_dataset.get_example_by_path(
        local_storage_file_path,
        int_label
    )

    return data


def _remove_example(index):
    if not _prefetch_multiprocess_iterator_terminating:
        path, _ = _pfetch_multiprocess_iterator_fetch_dataset[index]

        backend_storage_file_path = os.path.join(_pfetch_multiprocess_iterator_fetch_dataset.root, path)
        delete_file_path = _solve_local_storage_path(_prefetch_multiprocess_iterator_local_storage_base,
                                                     backend_storage_file_path)
        if os.path.exists(delete_file_path):
            try:
                os.remove(delete_file_path)
            except FileNotFoundError:
                pass
