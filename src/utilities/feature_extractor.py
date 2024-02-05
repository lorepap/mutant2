from collections import deque
import statistics

class FeatureExtractor:
    def __init__(self, feature_names, window_sizes=['10', '200', '1000']):
        self.feature_names = feature_names
        self.window_sizes = window_sizes
        self.features = {size: {stat: {} for stat in ['win', 'min', 'max', 'avg']} for size in self.window_sizes}

        # Initialize deques for each feature and each window size
        for window_size in self.window_sizes:
            self.features[window_size]['win'] = {feature: deque(maxlen=int(window_size)) for feature in feature_names}
            for stat in ['min', 'max', 'avg']:
                self.features[window_size][stat] = {feature: [] for feature in feature_names}

    def update(self, feature_values):
        # Update feature values for each window size
        for window_size in self.window_sizes:
            for feature, value in enumerate(feature_values):
                self.features[window_size]['win'][self.feature_names[feature]].append(value)

    def compute_statistics(self):
        # Compute min, max, and avg statistics for each feature and window size
        for window_size in self.window_sizes:
            for feature in self.feature_names:
                feature_data = self.features[window_size]['win'][feature]
                if feature_data:
                    self.features[window_size]['min'][feature] = round(min(feature_data), 5)
                    self.features[window_size]['max'][feature] = round(max(feature_data), 5)
                    self.features[window_size]['avg'][feature] = round(statistics.mean(feature_data), 5)

    def get_statistics(self):
        return self.features
