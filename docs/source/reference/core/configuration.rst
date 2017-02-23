.. _configuration:

Configuring Chainer
===================

..  currentmodule:: chainer

Chainer provides some global settings that affect the behavior of some functionalities.
Such settings can be configured using the *unified configuration system*.
The system provides a transparent way to manage the configuration for each process and for each thread.

The configuration is managed by two global objects: :data:`chainer.global_config` and :data:`chainer.config`.

- The :data:`global_config` object maintains the configuration shared in the Python process.
  This is an instance of the :class:`~chainer.configuration.GlobalConfig` class.
  It can be used just as a plain object, and users can freely set any attributes on it.
- The :data:`config` object, on the other hand, maintains the configuration for the current thread.
  This is an instance of the :class:`~chainer.configuration.LocalConfig` class.
  It behaves like a thread-local object, and any attribute modifications are only visible to the current thread.

If no value is set to :data:`config` for a given key, :data:`global_config` is transparently referred.
Thanks to this transparent lookup, users can always use :data:`config` to read any configuration so that the thread-local configuration is used if available and otherwise the default global setting is used.

The following entries of the configuration are currently provided by Chainer.
Some entries support environment variables to set the default values.
Note that the default values are set in the global config.

``chainer.config.debug``
   Debug mode flag.
   If it is ``True``, Chainer runs in the debug mode.
   See :ref:`debug` for more information of the debug mode.
   The default value is given by ``CHAINER_DEBUG`` environment variable (set to 0 or 1) if available, otherwise uses ``False``.
``chainer.config.enable_backprop``
   Flag to enable backpropagation support.
   If it is ``True``, the default behavior of :class:`Function` application to :class:`Variable` is non-volatile if all inputs have ``AUTO`` volatile flag.
   Otherwise, the default behavior is set to volatile mode that doesn't keep track of any function appplications to all :class:`Variable` s.
   The default value is ``True``.
``chainer.config.train``
   Training mode flag.
   If it is ``True``, Chainer runs in the training mode.
   Otherwise, it runs in the testing (evaluation) mode.
   The default value is ``True``.
``chainer.config.type_check``
   Type checking mode flag.
   If it is ``True``, Chainer checks the types (data types and shapes) of inputs on :class:`Function` applications.
   Otherwise, it skips type checking.
   The default value is given by ``CHAINER_TYPE_CHECK`` environment variable (set to 0 or 1) if available, otherwise uses ``True``.

Users can also define their own configurations.
There are two ways:

1. Use Chainer's configuration objects.
   In this case, **it is strongly recommended to prefix the name by "user_"** to avoid name conflicts with configurations introduced to Chainer in the future.
2. Use your own configuration objects.
   Users can define their own configuration objects using :class:`chainer.configuration.GlobalConfig` and :class:`chainer.configuration.LocalConfig`.
   In this case, there is no need to take care of the name conflicts.


.. admonition:: Example

   If you want to share a setting within the process, set an attribute to the global configuration.

   .. doctest::

      >>> chainer.global_config.user_my_setting = 123

   This value is automatically extracted by referring to the local config.

   .. doctest::

      >>> chainer.config.user_my_setting
      123

   If you set an attribute to the local configuration, the value is only visible to the current thread.

   .. doctest::

      >>> chainer.config.user_my_setting = 123

   We often want to temporarily modify the configuration for the current thread.
   It can be done by using :func:`using_config`.
   For example, if you only want to enable debug mode in a fragment of code, write as follows.

      >>> with chainer.using_config('debug', True):
      ...     ...  # code running in the debug mode

   We often want to switch to the test mode for an evaluation.
   This is also done in the same way.

      >>> with chainer.using_config('train', False):
      ...     ...  # code running in the test mode

   Note that :class:`~chainer.training.extensions.Evaluator` automatically switches to the test mode, and thus you do not need to manually switch in the loss function for the evaluation.

   You can also make your own code behave differently in training and test modes as follows.

   .. code-block:: python

      if chainer.config.train:
          ...  # code only running in the training mode
      else:
          ...  # code only running in the test mode


.. autodata:: global_config
.. autodata:: config
.. autofunction:: using_config

.. autoclass:: chainer.configuration.GlobalConfig
   :members:

.. autoclass:: chainer.configuration.LocalConfig
   :members:
