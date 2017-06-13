import mock

from chainer import training


def get_trainer_with_mock_updater(stop_trigger=(10, 'iteration'),
                                  update_core=None):
    """Returns a :class:`~chainer.training.Trainer` object with mock updater.

    The returned trainer can be used for testing the trainer itself and the
    extensions. A mock object is used as its updater. The update function set
    to the mock correctly increments the iteration counts (
    ``updater.iteration``), and thus you can write a test relying on it.

    Args:
        stop_trigger: Stop trigger of the trainer.
        update_core: A function object to mock ``updater.update_core`` method.

    Returns:
        Trainer object with a mock updater.

    """
    updater = mock.Mock()
    updater.get_all_optimizers.return_value = {}
    updater.iteration = 0
    updater.epoch_detail = 1
    updater.update_core = update_core or (lambda: None)

    def update():
        updater.update_core()
        updater.iteration += 1

    updater.update = update
    trainer = training.Trainer(updater, stop_trigger)
    return trainer
