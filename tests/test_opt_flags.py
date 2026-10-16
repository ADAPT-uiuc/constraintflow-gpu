import os
import unittest
from unittest import mock
from constraintflow.compiler.optimizations import opt_flags


class OptFlagsTest(unittest.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_every_flag_defaults_off(self):
        self.assertFalse(any(opt_flags.enabled(f) for f in opt_flags.FLAGS))

    @mock.patch.dict(os.environ, {'CF_DPE_OPTS': '1'}, clear=True)
    def test_umbrella_enables_every_flag(self):
        self.assertTrue(all(opt_flags.enabled(f) for f in opt_flags.FLAGS))

    @mock.patch.dict(os.environ, {'CF_DPE_OPTS': '1', 'CF_PATCH_FACTOR': '0'}, clear=True)
    def test_individual_flag_overrides_umbrella(self):
        self.assertFalse(opt_flags.enabled('CF_PATCH_FACTOR'))
        self.assertTrue(opt_flags.enabled('CF_CONV_SUM'))

    @mock.patch.dict(os.environ, {'CF_VIEW_ALIASES': '1'}, clear=True)
    def test_individual_flag_alone(self):
        self.assertTrue(opt_flags.enabled('CF_VIEW_ALIASES'))
        self.assertFalse(opt_flags.enabled('CF_PATCH_FACTOR'))


if __name__ == '__main__':
    unittest.main()
