import unittest

from chainer import trainer


class TestExtension(unittest.TestCase):

    def test_default_name(self):
        class MyExtension(trainer.Extension):
            pass

        ext = MyExtension()
        self.assertEqual(ext.default_name, 'MyExtension')

    def test_make_extension(self):
        @trainer.make_extension(trigger=(2, 'epoch'), default_name='my_ext',
                                priority=50, invoke_before_training=True)
        def my_extension(trainer):
            pass

        self.assertEqual(my_extension.trigger, (2, 'epoch'))
        self.assertEqual(my_extension.default_name, 'my_ext')
        self.assertEqual(my_extension.priority, 50)
        self.assertTrue(my_extension.invoke_before_training)

    def test_make_extension_default_values(self):
        @trainer.make_extension()
        def my_extension(trainer):
            pass

        self.assertEqual(my_extension.trigger, (1, 'iteration'))
        self.assertEqual(my_extension.default_name, 'my_extension')
        self.assertEqual(my_extension.priority, trainer.PRIORITY_READER)
        self.assertFalse(my_extension.invoke_before_training)
