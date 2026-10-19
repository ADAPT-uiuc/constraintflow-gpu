import unittest
from constraintflow.compiler.optimizations import opt_flags


class OptFlagsTest(unittest.TestCase):
    def tearDown(self):
        opt_flags.reset()

    def test_every_flag_defaults_on(self):
        self.assertTrue(all(opt_flags.enabled(f) for f in opt_flags.FLAGS))

    def test_set_enabled(self):
        opt_flags.set_enabled(['view_aliases'])
        self.assertTrue(opt_flags.enabled('view_aliases'))
        self.assertFalse(opt_flags.enabled('patch_factor'))
        opt_flags.reset()
        self.assertTrue(opt_flags.enabled('patch_factor'))

    def test_unknown_flag(self):
        with self.assertRaises(ValueError):
            opt_flags.set_enabled(['shared_gathers'])


if __name__ == '__main__':
    unittest.main()
