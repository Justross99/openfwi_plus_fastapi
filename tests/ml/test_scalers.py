import pytest
import numpy as np
from pathlib import Path
import shutil
from data.scalers import StandardScaler, MinMaxScaler

# --- Fixtures ---


@pytest.fixture
def dummy_npy_files():
    """Fixture to create dummy .npy files for scaler testing."""
    # Use a local temp path to avoid permission issues, mirroring test_dataloader.py
    tmp_path = Path("tests/ml/tmp_scaler_data")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    file_dir = tmp_path / "scaler_data"
    file_dir.mkdir(parents=True)

    f1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    f2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    path1 = file_dir / "file1.npy"
    path2 = file_dir / "file2.npy"

    np.save(path1, f1)
    np.save(path2, f2)

    yield [str(path1), str(path2)]

    # Teardown
    shutil.rmtree(tmp_path)


@pytest.fixture
def fitted_standard_scaler(dummy_npy_files):
    """Provides a StandardScaler instance fitted to the dummy files."""
    scaler = StandardScaler()
    scaler.fit(dummy_npy_files)
    return scaler


@pytest.fixture
def fitted_min_max_scaler(dummy_npy_files):
    """Provides a MinMaxScaler instance fitted to the dummy files."""
    scaler = MinMaxScaler()
    scaler.fit(dummy_npy_files)
    return scaler


# --- Tests for StandardScaler ---


def test_standard_scaler_fit(fitted_standard_scaler):
    # Mean of [1,2,3,4,5,6] is 3.5
    # Std dev is approx 1.7078
    assert fitted_standard_scaler.count == 6
    np.testing.assert_allclose(fitted_standard_scaler.mean, 3.5)
    np.testing.assert_allclose(fitted_standard_scaler.std, 1.707825, rtol=1e-5)


def test_standard_scaler_transform(fitted_standard_scaler):
    sample_data = np.array([1.0, 3.5, 6.0])
    transformed = fitted_standard_scaler.transform(sample_data)

    # Check that the mean is shifted to 0
    assert np.isclose(transformed[1], 0.0)
    # Check scaling
    expected = (sample_data - 3.5) / 1.707825
    np.testing.assert_allclose(transformed, expected, rtol=1e-5)


def test_standard_scaler_inverse_transform(fitted_standard_scaler):
    original_data = np.array([1.0, 3.5, 6.0])
    transformed = fitted_standard_scaler.transform(original_data)
    inversed = fitted_standard_scaler.inverse_transform(transformed)
    np.testing.assert_allclose(inversed, original_data)


def test_standard_scaler_unfitted_transform_raises_error():
    scaler = StandardScaler()
    with pytest.raises(RuntimeError, match="StandardScaler has not been fitted yet."):
        scaler.transform(np.array([1, 2]))


# --- Tests for MinMaxScaler ---


def test_min_max_scaler_fit(fitted_min_max_scaler):
    assert fitted_min_max_scaler.min_val == 1.0
    assert fitted_min_max_scaler.max_val == 6.0


def test_min_max_scaler_transform(fitted_min_max_scaler):
    sample_data = np.array([1.0, 3.5, 6.0])
    transformed = fitted_min_max_scaler.transform(sample_data)

    # 1.0 should be 0, 6.0 should be 1, 3.5 should be 0.5
    expected = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(transformed, expected)


def test_min_max_scaler_inverse_transform(fitted_min_max_scaler):
    original_data = np.array([1.0, 3.5, 6.0])
    transformed = fitted_min_max_scaler.transform(original_data)
    inversed = fitted_min_max_scaler.inverse_transform(transformed)
    np.testing.assert_allclose(inversed, original_data)


def test_min_max_scaler_unfitted_transform_raises_error():
    scaler = MinMaxScaler()
    with pytest.raises(RuntimeError, match="Scaler has not been fitted yet."):
        scaler.transform(np.array([1, 2]))
