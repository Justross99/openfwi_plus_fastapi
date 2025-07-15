import numpy as np


# --- MinMaxScaler for Normalizing Labels ---
class MinMaxScaler:
    """
    A simple Min-Max scaler to normalize data to a [0, 1] range.
    It can be fitted on a set of files and then used to transform data.
    """

    def __init__(self):
        self.min_val = np.inf
        self.max_val = -np.inf

    def fit(self, file_paths: list):
        """Find the min and max values across a list of .npy files."""
        for path in file_paths:
            try:
                chunk = np.load(path)
                self.min_val = min(self.min_val, np.min(chunk))
                self.max_val = max(self.max_val, np.max(chunk))
            except Exception as e:
                print(f"Warning: Could not process file {path} during scaling: {e}")
        print(f"Label Scaler fitted. Min: {self.min_val}, Max: {self.max_val}")

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale data to [0, 1] range."""
        if self.min_val == np.inf or self.max_val == -np.inf:
            raise RuntimeError("Scaler has not been fitted yet.")
        if self.max_val == self.min_val:
            return np.zeros_like(data)  # Avoid division by zero
        return (data - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Scale data from [0, 1] back to the original range."""
        if self.min_val == np.inf or self.max_val == -np.inf:
            raise RuntimeError("Scaler has not been fitted yet.")
        return data * (self.max_val - self.min_val) + self.min_val


# --- StandardScaler for Normalizing Inputs ---
class StandardScaler:
    """
    A simple StandardScaler to normalize data to have a mean of 0 and a std of 1.
    It fits on a set of files using a two-pass method for numerical stability.
    """

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.count = 0

    def fit(self, file_paths: list):
        """Find the mean and std dev across a list of .npy files."""
        # First pass: calculate the mean
        total_sum = 0.0
        total_count = 0
        for path in file_paths:
            try:
                chunk = np.load(path)
                total_sum += np.sum(chunk)
                total_count += np.size(chunk)
            except Exception as e:
                print(
                    f"Warning: Could not process file {path} during scaling pass 1: {e}"
                )

        if total_count == 0:
            print("Warning: No data found to fit StandardScaler.")
            return

        self.mean = total_sum / total_count
        self.count = total_count

        # Second pass: calculate the variance
        sum_of_squared_diff = 0.0
        for path in file_paths:
            try:
                chunk = np.load(path)
                sum_of_squared_diff += np.sum((chunk - self.mean) ** 2)
            except Exception as e:
                print(
                    f"Warning: Could not process file {path} during scaling pass 2: {e}"
                )

        variance = sum_of_squared_diff / self.count
        self.std = np.sqrt(variance)

        if self.std == 0:
            print("Warning: Standard deviation is zero. Scaling will result in zeros.")
            self.std = 1.0  # Avoid division by zero

        print(f"Input Scaler fitted. Mean: {self.mean}, Std: {self.std}")

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale data to have mean=0 and std=1."""
        if self.count == 0:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Scale data from normalized form back to the original scale."""
        if self.count == 0:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        return data * self.std + self.mean
