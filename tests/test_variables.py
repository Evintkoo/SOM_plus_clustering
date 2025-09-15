"""
Tests for the variables module.
"""
import pytest

# Import the variables module
import sys
sys.path.append('d:/Projects/SOM_plus_clustering')
from modules.variables import (
    INITIATION_METHOD_LIST,
    DISTANCE_METHOD_LIST,
    EVAL_METHOD_LIST,
    CLASSIFICATION_EVAL_METHOD_LIST
)


class TestVariableConstants:
    """Test the constant variables defined in the module."""
    
    def test_initiation_method_list_content(self):
        """Test that INITIATION_METHOD_LIST contains expected methods."""
        expected_methods = [
            "random", "kde", "kmeans", "kmeans++", "som++", 
            "zero", "he", "naive_sharding", "lecun", "lsuv"
        ]
        
        assert isinstance(INITIATION_METHOD_LIST, list)
        assert len(INITIATION_METHOD_LIST) == len(expected_methods)
        
        for method in expected_methods:
            assert method in INITIATION_METHOD_LIST
    
    def test_initiation_method_list_types(self):
        """Test that all initiation methods are strings."""
        assert all(isinstance(method, str) for method in INITIATION_METHOD_LIST)
        assert all(len(method) > 0 for method in INITIATION_METHOD_LIST)  # No empty strings
    
    def test_initiation_method_list_uniqueness(self):
        """Test that all initiation methods are unique."""
        assert len(INITIATION_METHOD_LIST) == len(set(INITIATION_METHOD_LIST))
    
    def test_distance_method_list_content(self):
        """Test that DISTANCE_METHOD_LIST contains expected methods."""
        expected_methods = ["euclidean", "cosine"]
        
        assert isinstance(DISTANCE_METHOD_LIST, list)
        assert len(DISTANCE_METHOD_LIST) == len(expected_methods)
        
        for method in expected_methods:
            assert method in DISTANCE_METHOD_LIST
    
    def test_distance_method_list_types(self):
        """Test that all distance methods are strings."""
        assert all(isinstance(method, str) for method in DISTANCE_METHOD_LIST)
        assert all(len(method) > 0 for method in DISTANCE_METHOD_LIST)
    
    def test_distance_method_list_uniqueness(self):
        """Test that all distance methods are unique."""
        assert len(DISTANCE_METHOD_LIST) == len(set(DISTANCE_METHOD_LIST))
    
    def test_eval_method_list_content(self):
        """Test that EVAL_METHOD_LIST contains expected methods."""
        expected_methods = [
            "davies_bouldin", "silhouette", "calinski_harabasz", "dunn",
            "bcubed_recall", "bcubed_precision", "all"
        ]
        
        assert isinstance(EVAL_METHOD_LIST, list)
        assert len(EVAL_METHOD_LIST) == len(expected_methods)
        
        for method in expected_methods:
            assert method in EVAL_METHOD_LIST
    
    def test_eval_method_list_types(self):
        """Test that all evaluation methods are strings."""
        assert all(isinstance(method, str) for method in EVAL_METHOD_LIST)
        assert all(len(method) > 0 for method in EVAL_METHOD_LIST)
    
    def test_eval_method_list_uniqueness(self):
        """Test that all evaluation methods are unique."""
        assert len(EVAL_METHOD_LIST) == len(set(EVAL_METHOD_LIST))
    
    def test_eval_method_list_has_all(self):
        """Test that EVAL_METHOD_LIST contains 'all' option."""
        assert "all" in EVAL_METHOD_LIST
    
    def test_classification_eval_method_list_content(self):
        """Test that CLASSIFICATION_EVAL_METHOD_LIST contains expected methods."""
        expected_methods = ["accuracy", "f1_score", "recall", "all"]
        
        assert isinstance(CLASSIFICATION_EVAL_METHOD_LIST, list)
        assert len(CLASSIFICATION_EVAL_METHOD_LIST) == len(expected_methods)
        
        for method in expected_methods:
            assert method in CLASSIFICATION_EVAL_METHOD_LIST
    
    def test_classification_eval_method_list_types(self):
        """Test that all classification evaluation methods are strings."""
        assert all(isinstance(method, str) for method in CLASSIFICATION_EVAL_METHOD_LIST)
        assert all(len(method) > 0 for method in CLASSIFICATION_EVAL_METHOD_LIST)
    
    def test_classification_eval_method_list_uniqueness(self):
        """Test that all classification evaluation methods are unique."""
        assert len(CLASSIFICATION_EVAL_METHOD_LIST) == len(set(CLASSIFICATION_EVAL_METHOD_LIST))
    
    def test_classification_eval_method_list_has_all(self):
        """Test that CLASSIFICATION_EVAL_METHOD_LIST contains 'all' option."""
        assert "all" in CLASSIFICATION_EVAL_METHOD_LIST


class TestVariableRelationships:
    """Test relationships between different variable lists."""
    
    def test_lists_are_independent(self):
        """Test that the different method lists are independent."""
        # No overlap between initiation and distance methods
        initiation_set = set(INITIATION_METHOD_LIST)
        distance_set = set(DISTANCE_METHOD_LIST)
        
        assert len(initiation_set.intersection(distance_set)) == 0
        
        # Evaluation lists may have some overlap (like "all")
        eval_set = set(EVAL_METHOD_LIST)
        classification_eval_set = set(CLASSIFICATION_EVAL_METHOD_LIST)
        
        # But they should be mostly different
        overlap = eval_set.intersection(classification_eval_set)
        assert len(overlap) <= 1  # Only "all" should overlap
    
    def test_all_option_consistency(self):
        """Test that 'all' option appears in evaluation lists."""
        assert "all" in EVAL_METHOD_LIST
        assert "all" in CLASSIFICATION_EVAL_METHOD_LIST
    
    def test_method_naming_conventions(self):
        """Test that method names follow consistent naming conventions."""
        # All methods should be lowercase
        all_methods = (INITIATION_METHOD_LIST + DISTANCE_METHOD_LIST + 
                      EVAL_METHOD_LIST + CLASSIFICATION_EVAL_METHOD_LIST)
        
        for method in all_methods:
            assert method.islower(), f"Method '{method}' should be lowercase"
            # Should not start or end with whitespace
            assert method.strip() == method, f"Method '{method}' has leading/trailing whitespace"
    
    def test_specific_method_presence(self):
        """Test that specific important methods are present."""
        # Initiation methods
        assert "random" in INITIATION_METHOD_LIST
        assert "kmeans++" in INITIATION_METHOD_LIST
        assert "som++" in INITIATION_METHOD_LIST
        
        # Distance methods
        assert "euclidean" in DISTANCE_METHOD_LIST
        assert "cosine" in DISTANCE_METHOD_LIST
        
        # Evaluation methods
        assert "silhouette" in EVAL_METHOD_LIST
        assert "davies_bouldin" in EVAL_METHOD_LIST
        
        # Classification evaluation methods
        assert "accuracy" in CLASSIFICATION_EVAL_METHOD_LIST
        assert "f1_score" in CLASSIFICATION_EVAL_METHOD_LIST


class TestVariableImmutability:
    """Test that variables maintain their expected properties."""
    
    def test_lists_are_not_empty(self):
        """Test that none of the lists are empty."""
        assert len(INITIATION_METHOD_LIST) > 0
        assert len(DISTANCE_METHOD_LIST) > 0
        assert len(EVAL_METHOD_LIST) > 0
        assert len(CLASSIFICATION_EVAL_METHOD_LIST) > 0
    
    def test_lists_contain_valid_strings(self):
        """Test that all lists contain valid string identifiers."""
        all_lists = [
            INITIATION_METHOD_LIST,
            DISTANCE_METHOD_LIST,
            EVAL_METHOD_LIST,
            CLASSIFICATION_EVAL_METHOD_LIST
        ]
        
        for method_list in all_lists:
            for method in method_list:
                assert isinstance(method, str)
                assert len(method) > 0
                assert not method.isspace()  # Not just whitespace
                # Should be valid Python identifier characters (mostly)
                assert method.replace('_', '').replace('+', '').isalnum()
    
    def test_expected_list_sizes(self):
        """Test that lists have expected minimum sizes."""
        assert len(INITIATION_METHOD_LIST) >= 5  # Should have several initialization methods
        assert len(DISTANCE_METHOD_LIST) >= 2   # Should have at least euclidean and cosine
        assert len(EVAL_METHOD_LIST) >= 4       # Should have several evaluation methods
        assert len(CLASSIFICATION_EVAL_METHOD_LIST) >= 3  # Should have several classification metrics


class TestVariableUsage:
    """Test how variables would be used in practice."""
    
    def test_initiation_method_validation(self):
        """Test validation patterns for initiation methods."""
        # Common validation pattern
        def is_valid_initiation_method(method):
            return method in INITIATION_METHOD_LIST
        
        # Test valid methods
        assert is_valid_initiation_method("random")
        assert is_valid_initiation_method("kmeans++")
        
        # Test invalid methods
        assert not is_valid_initiation_method("invalid_method")
        assert not is_valid_initiation_method("")
        assert not is_valid_initiation_method("RANDOM")  # Case sensitive
    
    def test_distance_method_validation(self):
        """Test validation patterns for distance methods."""
        def is_valid_distance_method(method):
            return method in DISTANCE_METHOD_LIST
        
        # Test valid methods
        assert is_valid_distance_method("euclidean")
        assert is_valid_distance_method("cosine")
        
        # Test invalid methods
        assert not is_valid_distance_method("manhattan")
        assert not is_valid_distance_method("euclidian")  # Common misspelling
    
    def test_eval_method_validation(self):
        """Test validation patterns for evaluation methods."""
        def is_valid_eval_method(method):
            return method in EVAL_METHOD_LIST
        
        # Test valid methods
        assert is_valid_eval_method("silhouette")
        assert is_valid_eval_method("all")
        
        # Test invalid methods
        assert not is_valid_eval_method("silhoutte")  # Common misspelling
        assert not is_valid_eval_method("ALL")  # Case sensitive
    
    def test_classification_eval_method_validation(self):
        """Test validation patterns for classification evaluation methods."""
        def is_valid_classification_eval_method(method):
            return method in CLASSIFICATION_EVAL_METHOD_LIST
        
        # Test valid methods
        assert is_valid_classification_eval_method("accuracy")
        assert is_valid_classification_eval_method("f1_score")
        
        # Test invalid methods
        assert not is_valid_classification_eval_method("precision")  # Not in list
        assert not is_valid_classification_eval_method("f1-score")   # Wrong format
    
    def test_method_subset_checking(self):
        """Test checking if a set of methods is valid."""
        def are_valid_eval_methods(methods):
            return set(methods).issubset(set(EVAL_METHOD_LIST))
        
        # Test valid subsets
        assert are_valid_eval_methods(["silhouette"])
        assert are_valid_eval_methods(["silhouette", "davies_bouldin"])
        assert are_valid_eval_methods([])  # Empty subset is valid
        
        # Test invalid subsets
        assert not are_valid_eval_methods(["silhouette", "invalid_method"])
        assert not are_valid_eval_methods(["precision", "recall"])  # Wrong list
    
    def test_all_method_filtering(self):
        """Test filtering out 'all' from method lists."""
        def get_specific_eval_methods():
            return [method for method in EVAL_METHOD_LIST if method != "all"]
        
        specific_methods = get_specific_eval_methods()
        assert "all" not in specific_methods
        assert len(specific_methods) == len(EVAL_METHOD_LIST) - 1
        assert all(method in EVAL_METHOD_LIST for method in specific_methods)


class TestVariableDocumentation:
    """Test that variables are properly documented/self-explanatory."""
    
    def test_method_names_are_descriptive(self):
        """Test that method names are reasonably descriptive."""
        # Initiation methods should indicate their purpose
        descriptive_init_methods = {
            "random", "kde", "kmeans", "kmeans++", "som++",
            "zero", "he", "naive_sharding", "lecun", "lsuv"
        }
        
        for method in INITIATION_METHOD_LIST:
            assert method in descriptive_init_methods, f"Unknown initiation method: {method}"
        
        # Distance methods should be recognizable
        assert "euclidean" in DISTANCE_METHOD_LIST
        assert "cosine" in DISTANCE_METHOD_LIST
        
        # Evaluation methods should be recognizable metrics
        known_eval_methods = {
            "davies_bouldin", "silhouette", "calinski_harabasz", "dunn",
            "bcubed_recall", "bcubed_precision", "all"
        }
        
        for method in EVAL_METHOD_LIST:
            assert method in known_eval_methods, f"Unknown evaluation method: {method}"
    
    def test_method_categories_are_coherent(self):
        """Test that methods in each category belong together."""
        # All initiation methods should be reasonable initialization strategies
        init_method_types = {
            "random": "random_initialization",
            "kde": "density_based",
            "kmeans": "clustering_based", 
            "kmeans++": "clustering_based",
            "som++": "distance_based",
            "zero": "structured",
            "he": "weight_initialization",
            "naive_sharding": "partitioning",
            "lecun": "weight_initialization", 
            "lsuv": "weight_initialization"
        }
        
        for method in INITIATION_METHOD_LIST:
            assert method in init_method_types, f"Uncategorized initiation method: {method}"
        
        # Distance methods should be valid distance functions
        assert len(DISTANCE_METHOD_LIST) >= 2  # At least euclidean and cosine
        
        # Evaluation methods should be clustering evaluation metrics
        clustering_metrics = {"davies_bouldin", "silhouette", "calinski_harabasz", "dunn"}
        unsupervised_metrics = {"bcubed_recall", "bcubed_precision"}
        
        for method in EVAL_METHOD_LIST:
            if method != "all":
                assert (method in clustering_metrics or 
                       method in unsupervised_metrics), f"Unknown metric type: {method}"