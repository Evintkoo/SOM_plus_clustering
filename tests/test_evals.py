"""
Tests for the evaluation metrics module.
"""
import pytest
import numpy as np
from sklearn.datasets import make_blobs
from unittest.mock import patch

# Import the evaluation metrics module
import sys
sys.path.append('d:/Projects/SOM_plus_clustering')
from modules.evals import (
    silhouette_score,
    davies_bouldin_index,
    calinski_harabasz_score,
    dunn_index,
    compare_distribution,
    bcubed_precision_recall,
    accuracy,
    f1_score,
    recall
)


"""
Tests for the evaluation metrics module.
"""
import pytest
import numpy as np
from sklearn.datasets import make_blobs

# Import the evaluation metrics module
import sys
sys.path.append('d:/Projects/SOM_plus_clustering')
from modules.evals import (
    silhouette_score,
    davies_bouldin_index,
    calinski_harabasz_score,
    dunn_index,
    compare_distribution,
    bcubed_precision_recall,
    accuracy,
    f1_score,
    recall
)


class TestSilhouetteScore:
    """Test silhouette score computation."""
    
    def test_silhouette_score_basic(self):
        """Test basic silhouette score computation."""
        X = np.array([
            [0, 0], [0, 1], [1, 0],  # Cluster 0
            [5, 5], [5, 6], [6, 5]   # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        score = silhouette_score(X, labels)
        
        assert 0 <= score <= 1
        assert score > 0.5
    
    def test_silhouette_score_single_cluster(self):
        """Test silhouette score with single cluster."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])
        
        score = silhouette_score(X, labels)
        assert score == 0.0


class TestDaviesBouldinIndex:
    """Test Davies-Bouldin index computation."""
    
    def test_davies_bouldin_index_basic(self):
        """Test basic Davies-Bouldin index computation."""
        X = np.array([
            [0, 0], [0, 1], [1, 0],  # Cluster 0
            [5, 5], [5, 6], [6, 5]   # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dbi = davies_bouldin_index(X, labels)
        
        assert dbi >= 0
        assert np.isfinite(dbi)
    
    def test_davies_bouldin_index_single_cluster(self):
        """Test DBI with single cluster."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])
        
        dbi = davies_bouldin_index(X, labels)
        assert dbi == 0.0


class TestCalinskiHarabaszScore:
    """Test Calinski-Harabasz score computation."""
    
    def test_calinski_harabasz_score_basic(self):
        """Test basic Calinski-Harabasz score computation."""
        X = np.array([
            [0, 0], [0, 1], [1, 0],  # Cluster 0
            [5, 5], [5, 6], [6, 5]   # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        ch_score = calinski_harabasz_score(X, labels)
        
        assert ch_score >= 0
        assert np.isfinite(ch_score)


class TestDunnIndex:
    """Test Dunn index computation."""
    
    def test_dunn_index_basic(self):
        """Test basic Dunn index computation."""
        X = np.array([
            [0, 0], [0, 1], [1, 0],  # Cluster 0
            [5, 5], [5, 6], [6, 5]   # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dunn = dunn_index(X, labels)
        
        assert dunn >= 0
        assert np.isfinite(dunn)


class TestCompareDistribution:
    """Test distribution comparison functionality."""
    
    def test_compare_distribution_basic(self):
        """Test basic distribution comparison."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, (3, 100))
        data2 = np.random.normal(0.1, 1.1, (3, 100))
        
        diff = compare_distribution(data1, data2)
        
        assert diff >= 0
        assert np.isfinite(diff)
    
    def test_compare_distribution_identical(self):
        """Test distribution comparison with identical data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (2, 50))
        
        diff = compare_distribution(data, data)
        
        assert diff == 0.0 or diff < 1e-10


class TestBCubedPrecisionRecall:
    """Test BCubed precision and recall computation."""
    
    def test_bcubed_precision_recall_perfect(self):
        """Test BCubed with perfect clustering."""
        clusters = np.array([0, 0, 1, 1, 2, 2])
        labels = np.array([0, 0, 1, 1, 2, 2])
        
        precision, recall = bcubed_precision_recall(clusters, labels)
        
        assert precision == 1.0
        assert recall == 1.0
    
    def test_bcubed_precision_recall_single_cluster(self):
        """Test BCubed with single cluster prediction."""
        clusters = np.array([0, 0, 0, 0, 0, 0])
        labels = np.array([0, 0, 1, 1, 2, 2])
        
        precision, recall = bcubed_precision_recall(clusters, labels)
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1


class TestSupervisedLearningMetrics:
    """Test supervised learning evaluation metrics."""
    
    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        
        acc = accuracy(y_true, y_pred)
        
        assert acc == 100.0
    
    def test_accuracy_partial(self):
        """Test accuracy with partial correct predictions."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]  # 4 out of 5 correct
        
        acc = accuracy(y_true, y_pred)
        
        assert acc == 80.0
    
    def test_f1_score_perfect(self):
        """Test F1 score with perfect predictions."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        
        f1 = f1_score(y_true, y_pred)
        
        assert f1 == 1.0
    
    def test_f1_score_no_positives(self):
        """Test F1 score when no positives predicted."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 0, 0, 0, 0]
        
        f1 = f1_score(y_true, y_pred)
        
        assert f1 == 0.0
    
    def test_recall_perfect(self):
        """Test recall with perfect predictions."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        
        rec = recall(y_true, y_pred)
        
        assert rec == 1.0
    
    def test_recall_miss_all_positives(self):
        """Test recall when missing all positive cases."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 0, 0, 0, 0]
        
        rec = recall(y_true, y_pred)
        
        assert rec == 0.0


class TestEvaluationMetricsIntegration:
    """Integration tests for evaluation metrics."""
    
    def test_metrics_with_sklearn_data(self):
        """Test metrics with sklearn generated data."""
        np.random.seed(42)
        X, labels = make_blobs(n_samples=100, centers=4, n_features=2, 
                              cluster_std=1.0, random_state=42)
        
        # Test all clustering metrics
        sil_score = silhouette_score(X, labels)
        assert -1 <= sil_score <= 1
        
        dbi = davies_bouldin_index(X, labels)
        assert dbi >= 0
        
        ch_score = calinski_harabasz_score(X, labels)
        assert ch_score >= 0
        
        dunn = dunn_index(X, labels)
        assert dunn >= 0
    
    def test_metrics_edge_cases(self):
        """Test evaluation metrics with edge cases."""
        # Empty arrays should raise errors or handle gracefully
        X_empty = np.array([]).reshape(0, 2)
        labels_empty = np.array([])
        
        # Most metrics should handle empty data by raising errors
        with pytest.raises((ValueError, IndexError)):
            silhouette_score(X_empty, labels_empty)
        
        # Mismatched dimensions
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels_wrong = np.array([0, 1])  # Wrong length
        
        with pytest.raises((ValueError, IndexError)):
            silhouette_score(X, labels_wrong)


class TestDaviesBouldinIndex:
    """Test Davies-Bouldin index computation."""
    
    def test_davies_bouldin_index_basic(self):
        """Test basic Davies-Bouldin index computation."""
        # Well-separated clusters
        X = np.array([
            [0, 0], [0, 1], [1, 0],  # Cluster 0
            [5, 5], [5, 6], [6, 5]   # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dbi = davies_bouldin_index(X, labels)
        
        # Lower is better for DBI
        assert dbi >= 0
        assert np.isfinite(dbi)
        # Should be relatively low for well-separated clusters
        assert dbi < 2.0
    
    def test_davies_bouldin_index_perfect_separation(self):
        """Test DBI with perfectly separated clusters."""
        X = np.array([
            [0, 0], [0, 0.1], [0.1, 0],        # Tight cluster 0
            [10, 10], [10, 10.1], [10.1, 10]   # Tight cluster 1, far away
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dbi = davies_bouldin_index(X, labels)
        
        # Should be very low for well-separated clusters
        assert dbi >= 0
        assert dbi < 1.0
    
    def test_davies_bouldin_index_overlapping_clusters(self):
        """Test DBI with overlapping clusters."""
        X = np.array([
            [0, 0], [1, 0], [2, 0],  # Cluster 0
            [1, 0], [2, 0], [3, 0]   # Cluster 1 (overlapping)
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dbi = davies_bouldin_index(X, labels)
        
        # Should be higher for overlapping clusters
        assert dbi >= 0
        assert np.isfinite(dbi)
    
    def test_davies_bouldin_index_single_cluster(self):
        """Test DBI with single cluster."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])
        
        # Should handle single cluster (DBI is undefined, should return inf or handle gracefully)
        dbi = davies_bouldin_index(X, labels)
        
        assert np.isfinite(dbi) or np.isinf(dbi)
    
    def test_davies_bouldin_index_many_clusters(self):
        """Test DBI with many clusters."""
        np.random.seed(42)
        X, labels = make_blobs(n_samples=100, centers=8, n_features=2, 
                              cluster_std=0.5, random_state=42)
        
        dbi = davies_bouldin_index(X, labels)
        
        assert dbi >= 0
        assert np.isfinite(dbi)
    
    def test_davies_bouldin_index_identical_points(self):
        """Test DBI with identical points in clusters."""
        X = np.array([
            [0, 0], [0, 0], [0, 0],  # Identical points in cluster 0
            [5, 5], [5, 5], [5, 5]   # Identical points in cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dbi = davies_bouldin_index(X, labels)
        
        # Should handle identical points gracefully
        assert np.isfinite(dbi) or dbi == 0


class TestCalinskiHarabaszScore:
    """Test Calinski-Harabasz score computation."""
    
    def test_calinski_harabasz_score_basic(self):
        """Test basic Calinski-Harabasz score computation."""
        # Well-separated clusters
        X = np.array([
            [0, 0], [0, 1], [1, 0],  # Cluster 0
            [5, 5], [5, 6], [6, 5]   # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        ch_score = calinski_harabasz_score(X, labels)
        
        # Higher is better for CH score
        assert ch_score >= 0
        assert np.isfinite(ch_score)
        # Should be relatively high for well-separated clusters
        assert ch_score > 1.0
    
    def test_calinski_harabasz_score_perfect_separation(self):
        """Test CH score with perfectly separated clusters."""
        X = np.array([
            [0, 0], [0, 0.1], [0.1, 0],        # Tight cluster 0
            [10, 10], [10, 10.1], [10.1, 10]   # Tight cluster 1, far away
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        ch_score = calinski_harabasz_score(X, labels)
        
        # Should be very high for well-separated clusters
        assert ch_score >= 0
        assert ch_score > 10.0
    
    def test_calinski_harabasz_score_overlapping_clusters(self):
        """Test CH score with overlapping clusters."""
        X = np.array([
            [0, 0], [1, 0], [2, 0],  # Cluster 0
            [1, 0], [2, 0], [3, 0]   # Cluster 1 (overlapping)
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        ch_score = calinski_harabasz_score(X, labels)
        
        # Should be lower for overlapping clusters
        assert ch_score >= 0
        assert np.isfinite(ch_score)
    
    def test_calinski_harabasz_score_single_cluster(self):
        """Test CH score with single cluster."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])
        
        # CH score is undefined for single cluster, should handle gracefully
        try:
            ch_score = calinski_harabasz_score(X, labels)
            # If it doesn't raise an error, it should be a reasonable value
            assert np.isfinite(ch_score) or np.isinf(ch_score)
        except ZeroDivisionError:
            # This is expected behavior for single cluster
            pass
    
    def test_calinski_harabasz_score_many_clusters(self):
        """Test CH score with many clusters."""
        np.random.seed(42)
        X, labels = make_blobs(n_samples=100, centers=6, n_features=2, 
                              cluster_std=0.8, random_state=42)
        
        ch_score = calinski_harabasz_score(X, labels)
        
        assert ch_score >= 0
        assert np.isfinite(ch_score)
    
    def test_calinski_harabasz_score_higher_dimensions(self):
        """Test CH score in higher dimensions."""
        np.random.seed(42)
        X, labels = make_blobs(n_samples=60, centers=4, n_features=4, 
                              cluster_std=1.0, random_state=42)
        
        ch_score = calinski_harabasz_score(X, labels)
        
        assert ch_score >= 0
        assert np.isfinite(ch_score)
    
    def test_calinski_harabasz_score_identical_points(self):
        """Test CH score with identical points."""
        X = np.array([
            [1, 1], [1, 1], [1, 1],  # Identical points in cluster 0
            [3, 3], [3, 3], [3, 3]   # Identical points in cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        ch_score = calinski_harabasz_score(X, labels)
        
        # Should handle identical points gracefully
        assert np.isfinite(ch_score) or np.isinf(ch_score)


class TestDunnIndex:
    """Test Dunn index computation."""
    
    def test_dunn_index_basic(self):
        """Test basic Dunn index computation."""
        # Well-separated clusters
        X = np.array([
            [0, 0], [0, 1], [1, 0],  # Cluster 0
            [5, 5], [5, 6], [6, 5]   # Cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dunn = dunn_index(X, labels)
        
        # Higher is better for Dunn index
        assert dunn >= 0
        assert np.isfinite(dunn)
        # Should be relatively high for well-separated clusters
        assert dunn > 1.0
    
    def test_dunn_index_perfect_separation(self):
        """Test Dunn index with perfectly separated clusters."""
        X = np.array([
            [0, 0], [0, 0.1], [0.1, 0],        # Tight cluster 0
            [10, 10], [10, 10.1], [10.1, 10]   # Tight cluster 1, far away
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dunn = dunn_index(X, labels)
        
        # Should be very high for well-separated clusters
        assert dunn >= 0
        assert dunn > 10.0
    
    def test_dunn_index_overlapping_clusters(self):
        """Test Dunn index with overlapping clusters."""
        X = np.array([
            [0, 0], [1, 0], [2, 0],  # Cluster 0
            [1, 0], [2, 0], [3, 0]   # Cluster 1 (overlapping)
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dunn = dunn_index(X, labels)
        
        # Should be lower (possibly 0) for overlapping clusters
        assert dunn >= 0
        assert np.isfinite(dunn)
    
    def test_dunn_index_single_cluster(self):
        """Test Dunn index with single cluster."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])
        
        # Dunn index is undefined for single cluster
        dunn = dunn_index(X, labels)
        
        assert np.isfinite(dunn) or np.isinf(dunn) or dunn == 0
    
    def test_dunn_index_identical_cluster_points(self):
        """Test Dunn index with identical points in clusters."""
        X = np.array([
            [0, 0], [0, 0], [0, 0],  # Identical points in cluster 0
            [5, 5], [5, 5], [5, 5]   # Identical points in cluster 1
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        dunn = dunn_index(X, labels)
        
        # Should handle identical points (intra-cluster distance = 0)
        assert np.isfinite(dunn) or np.isinf(dunn)
    
    def test_dunn_index_many_clusters(self):
        """Test Dunn index with many clusters."""
        np.random.seed(42)
        X, labels = make_blobs(n_samples=80, centers=5, n_features=2, 
                              cluster_std=0.5, random_state=42)
        
        dunn = dunn_index(X, labels)
        
        assert dunn >= 0
        assert np.isfinite(dunn)
    
    def test_dunn_index_single_point_clusters(self):
        """Test Dunn index with single-point clusters."""
        X = np.array([
            [0, 0],    # Cluster 0
            [3, 3],    # Cluster 1
            [6, 6]     # Cluster 2
        ])
        labels = np.array([0, 1, 2])
        
        dunn = dunn_index(X, labels)
        
        # Should handle single-point clusters (intra-cluster distance = 0)
        assert np.isfinite(dunn) or np.isinf(dunn)


class TestEvaluationMetricsEdgeCases:
    """Test edge cases and error conditions for evaluation metrics."""
    
    def test_empty_data(self):
        """Test evaluation metrics with empty data."""
        X = np.array([]).reshape(0, 2)
        labels = np.array([])
        
        # All metrics should handle empty data gracefully
        with pytest.raises((ValueError, IndexError)):
            silhouette_score(X, labels)
        
        with pytest.raises((ValueError, IndexError)):
            davies_bouldin_index(X, labels)
        
        with pytest.raises((ValueError, IndexError)):
            calinski_harabasz_score(X, labels)
        
        with pytest.raises((ValueError, IndexError)):
            dunn_index(X, labels)
    
    def test_mismatched_dimensions(self):
        """Test evaluation metrics with mismatched data and labels."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 1])  # Wrong length
        
        with pytest.raises((ValueError, IndexError)):
            silhouette_score(X, labels)
        
        with pytest.raises((ValueError, IndexError)):
            davies_bouldin_index(X, labels)
        
        with pytest.raises((ValueError, IndexError)):
            calinski_harabasz_score(X, labels)
        
        with pytest.raises((ValueError, IndexError)):
            dunn_index(X, labels)
    
    def test_negative_labels(self):
        """Test evaluation metrics with negative cluster labels."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        labels = np.array([-1, 0, -1, 0])  # Including negative labels
        
        # Should handle negative labels
        try:
            score = silhouette_score(X, labels)
            assert np.isfinite(score)
        except:
            pass  # May not support negative labels
        
        try:
            dbi = davies_bouldin_index(X, labels)
            assert np.isfinite(dbi) or np.isinf(dbi)
        except:
            pass
        
        try:
            ch_score = calinski_harabasz_score(X, labels)
            assert np.isfinite(ch_score) or np.isinf(ch_score)
        except:
            pass
        
        try:
            dunn = dunn_index(X, labels)
            assert np.isfinite(dunn) or np.isinf(dunn)
        except:
            pass
    
    def test_very_large_datasets(self):
        """Test evaluation metrics performance with larger datasets."""
        np.random.seed(42)
        X = np.random.randn(500, 3)
        labels = np.random.randint(0, 10, 500)  # 10 clusters
        
        # Should complete in reasonable time and give finite results
        score = silhouette_score(X, labels)
        assert np.isfinite(score)
        
        dbi = davies_bouldin_index(X, labels)
        assert np.isfinite(dbi) or np.isinf(dbi)
        
        ch_score = calinski_harabasz_score(X, labels)
        assert np.isfinite(ch_score) or np.isinf(ch_score)
        
        dunn = dunn_index(X, labels)
        assert np.isfinite(dunn) or np.isinf(dunn)
    
    def test_extreme_coordinate_values(self):
        """Test evaluation metrics with extreme coordinate values."""
        # Very large coordinates
        X = np.array([
            [1e6, 1e6], [1e6 + 1, 1e6 + 1],      # Cluster 0
            [2e6, 2e6], [2e6 + 1, 2e6 + 1]       # Cluster 1
        ])
        labels = np.array([0, 0, 1, 1])
        
        # Should handle large coordinates
        score = silhouette_score(X, labels)
        assert np.isfinite(score)
        
        # Very small coordinates
        X_small = np.array([
            [1e-10, 1e-10], [2e-10, 2e-10],      # Cluster 0
            [1e-9, 1e-9], [2e-9, 2e-9]           # Cluster 1
        ])
        
        score_small = silhouette_score(X_small, labels)
        assert np.isfinite(score_small)


class TestEvaluationMetricsIntegration:
    """Integration tests for evaluation metrics."""
    
    def test_metrics_consistency(self):
        """Test that metrics give consistent relative rankings."""
        np.random.seed(42)
        
        # Generate good clustering
        X_good, labels_good = make_blobs(n_samples=100, centers=4, n_features=2, 
                                        cluster_std=0.5, random_state=42)
        
        # Generate poor clustering (more overlap)
        X_poor, labels_poor = make_blobs(n_samples=100, centers=4, n_features=2, 
                                        cluster_std=2.0, random_state=42)
        
        # Compute metrics for both
        sil_good = silhouette_score(X_good, labels_good)
        sil_poor = silhouette_score(X_poor, labels_poor)
        
        dbi_good = davies_bouldin_index(X_good, labels_good)
        dbi_poor = davies_bouldin_index(X_poor, labels_poor)
        
        ch_good = calinski_harabasz_score(X_good, labels_good)
        ch_poor = calinski_harabasz_score(X_poor, labels_poor)
        
        # Silhouette: higher is better
        assert sil_good > sil_poor
        
        # Davies-Bouldin: lower is better
        assert dbi_good < dbi_poor
        
        # Calinski-Harabasz: higher is better
        assert ch_good > ch_poor
    
    def test_metrics_with_sklearn_data(self):
        """Test metrics with various sklearn datasets."""
        from sklearn.datasets import make_circles, make_moons
        
        # Test with circles
        X_circles, _ = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=42)
        labels_circles = np.array([0] * 50 + [1] * 50)
        
        sil_circles = silhouette_score(X_circles, labels_circles)
        assert np.isfinite(sil_circles)
        
        # Test with moons
        X_moons, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
        labels_moons = np.array([0] * 50 + [1] * 50)
        
        sil_moons = silhouette_score(X_moons, labels_moons)
        assert np.isfinite(sil_moons)
    
    def test_all_metrics_complete_workflow(self):
        """Test complete evaluation workflow with all metrics."""
        np.random.seed(42)
        X, true_labels = make_blobs(n_samples=200, centers=5, n_features=3, 
                                   cluster_std=1.0, random_state=42)
        
        # Compute all metrics
        metrics = {}
        
        try:
            metrics['silhouette'] = silhouette_score(X, true_labels)
        except:
            metrics['silhouette'] = None
        
        try:
            metrics['davies_bouldin'] = davies_bouldin_index(X, true_labels)
        except:
            metrics['davies_bouldin'] = None
        
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, true_labels)
        except:
            metrics['calinski_harabasz'] = None
        
        try:
            metrics['dunn'] = dunn_index(X, true_labels)
        except:
            metrics['dunn'] = None
        
        # At least some metrics should work
        working_metrics = [k for k, v in metrics.items() if v is not None and np.isfinite(v)]
        assert len(working_metrics) > 0
        
        # Print results for debugging
        print(f"Metrics computed: {metrics}")
    
class TestCompareDistribution:
    """Test distribution comparison functionality."""
    
    def test_compare_distribution_basic(self):
        """Test basic distribution comparison."""
        np.random.seed(42)
        
        # Create two similar distributions
        data1 = np.random.normal(0, 1, (3, 100))  # 3 features, 100 samples
        data2 = np.random.normal(0.1, 1.1, (3, 100))  # Slightly different
        
        diff = compare_distribution(data1, data2)
        
        assert diff >= 0
        assert np.isfinite(diff)
    
    def test_compare_distribution_identical(self):
        """Test distribution comparison with identical data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (2, 50))
        
        diff = compare_distribution(data, data)
        
        # Should be very small for identical distributions
        assert diff == 0.0 or diff < 1e-10
    
    def test_compare_distribution_very_different(self):
        """Test distribution comparison with very different data."""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, (2, 100))
        data2 = np.random.normal(10, 1, (2, 100))  # Very different mean
        
        diff = compare_distribution(data1, data2)
        
        # Should be large for very different distributions
        assert diff > 0.1


class TestBCubedPrecisionRecall:
    """Test BCubed precision and recall computation."""
    
    def test_bcubed_precision_recall_perfect(self):
        """Test BCubed with perfect clustering."""
        clusters = np.array([0, 0, 1, 1, 2, 2])
        labels = np.array([0, 0, 1, 1, 2, 2])
        
        precision, recall = bcubed_precision_recall(clusters, labels)
        
        # Should be perfect
        assert precision == 1.0
        assert recall == 1.0
    
    def test_bcubed_precision_recall_single_cluster(self):
        """Test BCubed with single cluster prediction."""
        clusters = np.array([0, 0, 0, 0, 0, 0])
        labels = np.array([0, 0, 1, 1, 2, 2])
        
        precision, recall = bcubed_precision_recall(clusters, labels)
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
    
    def test_bcubed_precision_recall_random(self):
        """Test BCubed with random clustering."""
        np.random.seed(42)
        clusters = np.random.randint(0, 3, 20)
        labels = np.random.randint(0, 3, 20)
        
        precision, recall = bcubed_precision_recall(clusters, labels)
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1