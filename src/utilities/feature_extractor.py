from collections import deque
import statistics

class FeatureExtractor:
    def __init__(self, feature_names, window_sizes=[10, 200, 1000]):
        self.feature_names = feature_names
        self.window_sizes = window_sizes
        self.features = {size: {stat: {} for stat in ['win', 'min', 'max', 'avg']} for size in window_sizes}

        # Initialize deques for each feature and each window size
        for window_size in window_sizes:
            self.features[window_size]['win'] = {feature: deque(maxlen=window_size) for feature in feature_names}
            for stat in ['min', 'max', 'avg']:
                self.features[window_size]['win'] = {feature: [] for feature in feature_names}

    def update(self, feature_values):
        # Update feature values for each window size
        for window_size in self.window_sizes:
            for feature, value in enumerate(feature_values):
                self.features[window_size]['avg'][self.feature_names[feature]].append(value)

    def compute_statistics(self):
        # Compute min, max, and avg statistics for each feature and window size
        for window_size in self.window_sizes:
            for feature in self.feature_names:
                feature_data = self.features[window_size]['win'][feature]
                if feature_data:
                    self.features[window_size]['min'][feature] = min(feature_data)
                    self.features[window_size]['max'][feature] = max(feature_data)
                    self.features[window_size]['avg'][feature] = statistics.mean(feature_data)

    def get_statistics(self):
        return self.features
