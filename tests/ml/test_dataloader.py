# filepath: tests/ml/test_dataloader.py
import pytest
import os
import numpy as np
from pathlib import Path
import shutil
from data.dataloader import (
    _get_id_from_filename,
    npy_chunk_generator,
    create_hf_dataset,
)
from data.scalers import StandardScaler, MinMaxScaler

# Constants for sample shapes, matching those in dataloader.py
DATA_SAMPLE_SHAPE = (5, 1000, 70)
LABEL_SAMPLE_SHAPE = (1, 70, 70)


@pytest.fixture
def dummy_npy_dirs():
    """Fixture to create dummy directories with matched and unmatched npy files."""
    # Use a local temp path to avoid permission issues
    tmp_path = Path("tests/ml/tmp_dataloader_data")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    data_dir = tmp_path / "data"
    label_dir = tmp_path / "labels"
    data_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # Matched pair 1: 3 samples
    d1 = np.ones((3, *DATA_SAMPLE_SHAPE), dtype=np.float32) * 10
    l1 = np.ones((3, *LABEL_SAMPLE_SHAPE), dtype=np.float32) * 1
    np.save(data_dir / "seis1.npy", d1)
    np.save(label_dir / "vel1.npy", l1)

    # Matched pair 2: 2 samples
    d2 = np.ones((2, *DATA_SAMPLE_SHAPE), dtype=np.float32) * 20
    l2 = np.ones((2, *LABEL_SAMPLE_SHAPE), dtype=np.float32) * 2
    np.save(data_dir / "data2.npy", d2)  # Using different prefix
    np.save(label_dir / "model2.npy", l2)

    # Matched pair 3 (for validation split): 4 samples
    d3 = np.ones((4, *DATA_SAMPLE_SHAPE), dtype=np.float32) * 30
    l3 = np.ones((4, *LABEL_SAMPLE_SHAPE), dtype=np.float32) * 3
    np.save(data_dir / "seis3.npy", d3)
    np.save(label_dir / "vel3.npy", l3)

    # Unmatched data file
    d_unmatched = np.random.rand(1, *DATA_SAMPLE_SHAPE).astype(np.float32)
    np.save(data_dir / "unmatched4.npy", d_unmatched)

    # Unmatched label file
    l_unmatched = np.random.rand(1, *LABEL_SAMPLE_SHAPE).astype(np.float32)
    np.save(label_dir / "unmatched5.npy", l_unmatched)

    yield str(data_dir), str(label_dir)

    # Teardown: remove the directory
    shutil.rmtree(tmp_path)


def test_get_id_from_filename():
    assert _get_id_from_filename("seis123.npy", ("seis",)) == "123"
    assert _get_id_from_filename("data45.npy", ("seis", "data")) == "45"
    assert _get_id_from_filename("vel_123_abc.npy", ("vel", "model")) == "_123_abc"
    assert _get_id_from_filename("model-45.npy", ("vel", "model")) == "-45"
    assert _get_id_from_filename("no_id.npy", ("seis", "data")) is None


def test_npy_chunk_generator_yields_correct_data(dummy_npy_dirs):
    data_dir, label_dir = dummy_npy_dirs

    data_files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
    label_files = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir))]

    # Manually pair them for the test
    paired_data_files = [f for f in data_files if "unmatched" not in f]
    paired_label_files = [f for f in label_files if "unmatched" not in f]

    generator = npy_chunk_generator(
        data_files_to_load=paired_data_files, label_files_to_load=paired_label_files
    )

    count = 0
    for item in generator:
        count += 1
        assert "data" in item
        assert "label" in item
        assert item["data"].shape == DATA_SAMPLE_SHAPE
        assert item["label"].shape == LABEL_SAMPLE_SHAPE

    # Total samples: 3 (from seis1) + 2 (from data2) + 4 (from seis3) = 9
    assert count == 9


def test_create_hf_dataset_train_val_split(dummy_npy_dirs):
    data_dir, label_dir = dummy_npy_dirs
    val_file_ratio = 0.3  # Corresponds to roughly 33% of 3 files -> 1 file for val

    # We have 3 files, so validation should get 1 file and train should get 2
    _, train_num_samples = create_hf_dataset(
        data_dir, label_dir, split="train", val_file_ratio=val_file_ratio, seed=42
    )
    _, val_num_samples = create_hf_dataset(
        data_dir, label_dir, split="validation", val_file_ratio=val_file_ratio, seed=42
    )

    # Total samples = 9. Train files have 3+2=5 samples. Val file has 4 samples.
    # The split is by file, not by sample.
    # With seed=42, the shuffled file order is 'seis1', 'seis3', 'data2'.
    # With val_file_ratio=0.3, num_val_files is 1.
    # Validation set: 'seis1.npy' (3 samples).
    # Training set: 'seis3.npy' (4 samples) and 'data2.npy' (2 samples).
    assert train_num_samples == 6
    assert val_num_samples == 3


def test_create_hf_dataset_with_scalers(dummy_npy_dirs):
    data_dir, label_dir = dummy_npy_dirs
    data_scaler = StandardScaler()
    label_scaler = MinMaxScaler()

    # Based on the split from the previous test (seed=42), the training files are
    # 'seis3.npy' (4 samples, value 30) and 'data2.npy' (2 samples, value 20).
    train_files = [
        os.path.join(data_dir, "seis3.npy"),
        os.path.join(data_dir, "data2.npy"),
    ]
    label_files = [
        os.path.join(label_dir, "vel3.npy"),
        os.path.join(label_dir, "model2.npy"),
    ]

    data_scaler.fit(train_files)
    label_scaler.fit(label_files)

    assert data_scaler.count > 0
    assert data_scaler.mean != 0.0
    # (4*30 + 2*20) / 6 samples = 160 / 6
    np.testing.assert_allclose(data_scaler.mean, 160.0 / 6.0)

    assert label_scaler.min_val != np.inf
    assert label_scaler.max_val != -np.inf
    assert label_scaler.min_val == 2.0
    assert label_scaler.max_val == 3.0

    # Create dataset with the pre-fitted scalers
    generator, num_samples = create_hf_dataset(
        data_dir,
        label_dir,
        split="train",
        val_file_ratio=0.3,
        seed=42,
        input_scaler=data_scaler,
        label_scaler=label_scaler,
    )

    # Check a sample from the dataset to ensure it was transformed
    first_batch = next(iter(generator()))
    # The first file in the training set is 'seis3.npy' (value 30)
    original_first_val = 30.0
    transformed_first_val = first_batch["data"][0][0][0][0]  # Get a single value
    assert not np.isclose(original_first_val, transformed_first_val)
