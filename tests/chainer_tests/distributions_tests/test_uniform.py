from chainer import distributions
from chainer import testing
import numpy


@testing.parameterize(*testing.product({
    'shape': [(2, 3), ()],
    'is_variable': [True, False],
    'sample_shape': [(3, 2), ()],
}))
@testing.fix_random()
@testing.with_requires('scipy')
class TestUniform(testing.distribution_unittest):

    scipy_onebyone = True

    def setUp_configure(self):
        from scipy import stats
        self.dist = distributions.Uniform
        self.scipy_dist = stats.uniform

        self.test_targets = set([
            "batch_shape", "cdf", "entropy", "event_shape", "icdf", "log_prob",
            "mean", "sample", "stddev", "support", "variance"])

        low = numpy.random.uniform(-10, 0, self.shape).astype(numpy.float32)
        high = numpy.random.uniform(
            low, low + 10, self.shape).astype(numpy.float32)
        self.params = {"low": low, "high": high}
        self.scipy_params = {"loc": low, "scale": high-low}

        self.support = '[low, high]'

    def sample_for_test(self):
        smp = numpy.random.normal(
            size=self.sample_shape + self.shape).astype(numpy.float32)
        return smp


testing.run_module(__name__, __file__)
