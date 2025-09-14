import unittest
import numpy as np

try:
    import cupy as cp  # noqa: F401
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

from modules.som import SOM


@unittest.skipUnless(CUPY_AVAILABLE, "CuPy not available; GPU tests skipped")
class TestSOMBasic(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n = 500
        self.dim = 8
        self.X = self.rng.normal(size=(self.n, self.dim)).astype(np.float32)

    def test_fit_predict_euclidean(self):
        som = SOM(m=5, n=6, dim=self.dim,
                  initiate_method="random",
                  learning_rate=0.5,
                  neighbour_rad=3,
                  distance_function="euclidean",
                  max_iter=np.inf)
        som.fit(self.X, epoch=2, shuffle=True)
        labels = som.predict(self.X)
        self.assertEqual(labels.shape[0], self.n)
        # labels are linear indices in [0, m*n)
        self.assertTrue(labels.min() >= 0)
        self.assertTrue(labels.max() < 5 * 6)

    def test_fit_predict_cosine(self):
        som = SOM(m=4, n=4, dim=self.dim,
                  initiate_method="random",
                  learning_rate=0.3,
                  neighbour_rad=2,
                  distance_function="cosine",
                  max_iter=np.inf)
        som.fit(self.X, epoch=1, shuffle=False)
        labels = som.predict(self.X)
        self.assertEqual(labels.shape[0], self.n)

    def test_cluster_centers_shape(self):
        som = SOM(m=3, n=7, dim=self.dim,
                  initiate_method="random",
                  learning_rate=0.4,
                  neighbour_rad=2,
                  distance_function="euclidean",
                  max_iter=np.inf)
        som.fit(self.X, epoch=1)
        centers = som.cluster_center_
        self.assertEqual(centers.shape, (3 * 7, self.dim))

    def test_evaluate_all(self):
        som = SOM(m=4, n=5, dim=self.dim,
                  initiate_method="random",
                  learning_rate=0.5,
                  neighbour_rad=3,
                  distance_function="euclidean",
                  max_iter=np.inf)
        som.fit(self.X, epoch=1)
        scores = som.evaluate(self.X, method=["all"])
        # Expect dictionary with known keys
        self.assertIn("silhouette", scores)
        self.assertIn("davies_bouldin", scores)
        self.assertIn("calinski_harabasz", scores)
        self.assertIn("dunn", scores)


if __name__ == "__main__":
    unittest.main()
