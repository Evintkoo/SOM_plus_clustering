import unittest

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

from modules.utils import one_hot_encode


@unittest.skipUnless(CUPY_AVAILABLE, "CuPy not available; GPU tests skipped")
class TestOneHot(unittest.TestCase):
    def test_one_hot_simple(self):
        y = cp.asarray([2, 0, 1, 2, 1, 0])
        enc = one_hot_encode(y)
        self.assertEqual(enc.shape, (6, 3))
        # Check rows
        self.assertTrue((enc[0] == cp.asarray([0, 0, 1])).all())
        self.assertTrue((enc[1] == cp.asarray([1, 0, 0])).all())
        self.assertTrue((enc[2] == cp.asarray([0, 1, 0])).all())

    def test_one_hot_non_contiguous_labels(self):
        y = cp.asarray([10, 20, 10, 30])
        enc = one_hot_encode(y)
        # Unique labels are [10,20,30]
        self.assertEqual(enc.shape, (4, 3))
        self.assertTrue((enc[0] == cp.asarray([1, 0, 0])).all())
        self.assertTrue((enc[1] == cp.asarray([0, 1, 0])).all())
        self.assertTrue((enc[3] == cp.asarray([0, 0, 1])).all())


if __name__ == "__main__":
    unittest.main()
