import unittest
import xarray as xr
import numpy as np
from preproc.preproc import InSituPreprocessor

class TestInSituPreprocessor(unittest.TestCase):

    def setUp(self):
        # Create a dummy xarray dataset with the necessary grid information
        self.grid_noblank_ds = xr.Dataset(
            coords={
                'XC': (['x', 'y'], np.random.rand(10, 10)),  # 10x10 grid for XC
                'YC': (['x', 'y'], np.random.rand(10, 10))   # 10x10 grid for YC
            }
        )

    def test_invalid_pkg(self):
        # Test invalid package type
        with self.assertRaises(ValueError) as context:
            InSituPreprocessor(pkg='invalid_pkg', grid_noblank_ds=self.grid_noblank_ds)
        self.assertIn("Invalid pkg 'invalid_pkg'. Must be either 'profiles' or 'obsfit'.", str(context.exception))

    def test_missing_xc_yc_in_grid(self):
        # Create dataset missing 'XC' and 'YC' to test error handling
        grid_without_xc_yc = xr.Dataset(coords={})

        with self.assertRaises(ValueError) as context:
            InSituPreprocessor(pkg='profiles', grid_noblank_ds=grid_without_xc_yc)
        self.assertIn("grid_noblank_ds must have fields XC and YC.", str(context.exception))

    def test_valid_initialization(self):
        # Test valid initialization with correct inputs
        processor = InSituPreprocessor(pkg='profiles', grid_noblank_ds=self.grid_noblank_ds)

        # Check that all attributes are correctly set
        self.assertEqual(processor.pkg, 'profiles')
        self.assertEqual(processor.grid, 'sphericalpolar')
        self.assertEqual(processor.sNx, 30)
        self.assertEqual(processor.sNy, 30)
        self.assertTrue(np.array_equal(processor.xc, self.grid_noblank_ds.XC.values))
        self.assertTrue(np.array_equal(processor.yc, self.grid_noblank_ds.YC.values))

if __name__ == '__main__':
    unittest.main()
