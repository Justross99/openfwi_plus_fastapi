import os
import numpy as np
from pathlib import Path
from datasets import Dataset, Features, Array3D
import jax.numpy as jnp
from .scalers import StandardScaler, MinMaxScaler


def _get_id_from_filename(filename: str, prefixes: tuple[str, ...]) -> str | None:
    """
    Extracts the ID from a filename by stripping known prefixes.
    Example: _get_id_from_filename("seis123.npy", ("seis", "data")) -> "123"
    """
    for prefix in prefixes:
        if filename.startswith(prefix):
            # e.g., 'seis123.npy' -> '123.npy' -> '123'
            return os.path.splitext(filename[len(prefix) :])[0]
    return None


def npy_chunk_generator(
    data_files_to_load: list,
    label_files_to_load: list,
    input_scaler: StandardScaler | None = None,
    label_scaler: MinMaxScaler | None = None,
):
    """
    Generator that yields dicts of {'data': ..., 'label': ...} for each sample.
    Processes a provided list of data and label .npy files.
    Each .npy file contains a chunk of samples.
    Applies scaling to inputs and labels if scalers are provided.
    """
    if len(data_files_to_load) != len(label_files_to_load):
        raise ValueError(
            "Mismatch in the number of data and label files provided to npy_chunk_generator."
        )

    for data_path, label_path in zip(data_files_to_load, label_files_to_load):
        try:
            data_chunk = np.load(data_path)
            label_chunk = np.load(label_path)
        except Exception as e:
            print(f"Warning: Error loading file {data_path} or {label_path}: {e}")
            continue

        if data_chunk.shape[0] != label_chunk.shape[0]:
            print(
                f"Warning: Mismatch in number of samples in chunk: {data_path} ({data_chunk.shape[0]}) vs {label_path} ({label_chunk.shape[0]})"
            )
            continue
        for d, l in zip(data_chunk, label_chunk):
            data_to_yield = d.astype(np.float32)
            if input_scaler:
                data_to_yield = input_scaler.transform(data_to_yield)

            label_to_yield = l.astype(np.float32)
            if label_scaler:
                label_to_yield = label_scaler.transform(label_to_yield)
            yield {"data": data_to_yield, "label": label_to_yield}


def create_hf_dataset(
    data_dir: str,
    label_dir: str,
    split: str = "train",
    val_file_ratio: float = 0.2,
    shuffle_files_before_split: bool = True,
    seed: int = 42,
    batch_size: int = 16,
    shuffle_dataset_items: bool = True,
    num_proc: int = 4,
    input_scaler: StandardScaler | None = None,  # Accept optional pre-fitted scaler
    label_scaler: MinMaxScaler | None = None,  # Accept optional pre-fitted scaler
):
    """
    Create a Hugging Face Dataset from .npy chunked files.
    Supports splitting files into training and validation sets.
    Returns a tuple: (batch_iterator, num_samples)
    batch_iterator yields (data, label) NumPy batches.
    num_samples is the total number of individual samples in this dataset split.
    """
    data_prefixes = ("seis", "data")
    label_prefixes = ("vel", "model")

    all_data_filenames = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    all_label_filenames = [f for f in os.listdir(label_dir) if f.endswith(".npy")]

    data_id_map = {
        _get_id_from_filename(fname, data_prefixes): os.path.join(data_dir, fname)
        for fname in all_data_filenames
        if _get_id_from_filename(fname, data_prefixes) is not None
    }

    label_id_map = {
        _get_id_from_filename(fname, label_prefixes): os.path.join(label_dir, fname)
        for fname in all_label_filenames
        if _get_id_from_filename(fname, label_prefixes) is not None
    }

    common_ids = sorted(list(set(data_id_map.keys()) & set(label_id_map.keys())))

    if not common_ids:
        raise ValueError(
            f"No matching file IDs found between data_dir ({data_dir}) and label_dir ({label_dir}). "
            f"Check prefixes {data_prefixes}, {label_prefixes} and file names."
        )

    paired_files = [(data_id_map[id], label_id_map[id]) for id in common_ids]

    if shuffle_files_before_split:
        rng = np.random.RandomState(seed)
        rng.shuffle(paired_files)

    num_total_files = len(paired_files)
    num_val_files = int(np.floor(num_total_files * val_file_ratio))

    if val_file_ratio > 0 and num_val_files == 0 and num_total_files > 0:
        num_val_files = (
            1  # Ensure at least one validation file if ratio > 0 and files exist
        )
    if num_val_files >= num_total_files:  # Ensure at least one training file
        num_val_files = num_total_files - 1

    if split == "train":
        # Files from num_val_files to the end are for training
        selected_paired_files = paired_files[num_val_files:]
        # shuffle_dataset_items is typically True for training
    elif split == "validation":
        # First num_val_files are for validation
        selected_paired_files = paired_files[:num_val_files]
        shuffle_dataset_items = (
            False  # Typically, validation set is not shuffled during iteration
        )
    else:
        raise ValueError(f"split must be 'train' or 'validation', got {split}")

    if not selected_paired_files:
        print(
            f"Warning: No files selected for split '{split}'. Check val_file_ratio and number of files."
        )

        # Return an empty iterator or handle as appropriate
        def empty_generator():
            yield from ()

        return empty_generator, 0  # Return 0 samples for empty iterator

    selected_data_files = [pair[0] for pair in selected_paired_files]
    selected_label_files = [pair[1] for pair in selected_paired_files]

    # Define features for efficient Arrow storage - these shapes must match your data
    # Consider making these configurable if datasets vary in shape
    features = Features(
        {
            "data": Array3D(shape=(5, 1000, 70), dtype="float32"),
            "label": Array3D(shape=(1, 70, 70), dtype="float32"),
        }
    )

    # The generator function needs to be defined here to capture the variables
    def generator():
        yield from npy_chunk_generator(
            selected_data_files, selected_label_files, input_scaler, label_scaler
        )

    ds = Dataset.from_generator(
        generator,
        features=features,
    )

    num_samples = len(ds)  # Get the number of samples

    if shuffle_dataset_items:
        ds = ds.shuffle(seed=seed)  # Shuffle samples within the dataset

    ds = ds.with_format("numpy")

    def batch_iterator():
        if len(ds) == 0:
            yield from ()  # Return empty generator if dataset is empty
            return
        for i in range(0, len(ds), batch_size):
            batch = ds[i : i + batch_size]
            # Ensure batch is not empty (can happen if len(ds) % batch_size != 0 for last batch)
            if batch["data"].shape[0] > 0:
                yield batch

    return batch_iterator, num_samples


def create_hf_dataset_jax_wrapper(
    data_dir: str,
    label_dir: str,
    val_file_ratio: float = 0.2,
    shuffle_files_before_split: bool = True,
    seed: int = 42,
    batch_size: int = 16,
    num_proc: int = 4,
    input_scaler: StandardScaler | None = None,  # Accept optional scaler
    label_scaler: MinMaxScaler | None = None,  # Accept optional scaler
):
    """
    Wrapper that creates and returns JAX-ified train and validation data loaders.
    This version now includes fitting a scaler on the training labels and applying it.
    Returns a tuple: (train_loader, num_train_samples, val_loader, num_val_samples, input_scaler, label_scaler)
    """
    # --- 1. File Splitting Logic (to get training file lists) ---
    data_prefixes = ("seis", "data")
    label_prefixes = ("vel", "model")
    all_data_filenames = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
    all_label_filenames = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
    data_id_map = {
        _get_id_from_filename(fname, data_prefixes): os.path.join(data_dir, fname)
        for fname in all_data_filenames
        if _get_id_from_filename(fname, data_prefixes) is not None
    }
    label_id_map = {
        _get_id_from_filename(fname, label_prefixes): os.path.join(label_dir, fname)
        for fname in all_label_filenames
        if _get_id_from_filename(fname, label_prefixes) is not None
    }
    common_ids = sorted(list(set(data_id_map.keys()) & set(label_id_map.keys())))
    if not common_ids:
        raise ValueError(
            f"No matching file IDs found between {data_dir} and {label_dir}."
        )

    paired_files = [(data_id_map[id], label_id_map[id]) for id in common_ids]
    if shuffle_files_before_split:
        rng = np.random.RandomState(seed)
        rng.shuffle(paired_files)

    num_total_files = len(paired_files)
    num_val_files = int(np.floor(num_total_files * val_file_ratio))
    if val_file_ratio > 0 and num_val_files == 0 and num_total_files > 0:
        num_val_files = 1

    if num_val_files >= num_total_files:  # Ensure at least one training file
        num_val_files = num_total_files - 1

    train_files = paired_files[num_val_files:]
    train_input_files = [pair[0] for pair in train_files]
    train_label_files = [pair[1] for pair in train_files]

    # --- 2. Fit the Scalers on the TRAINING data files if not provided ---
    if input_scaler is None:
        print("Fitting input scaler on training data...")
        input_scaler = StandardScaler()
        input_scaler.fit(train_input_files)
    else:
        print("Using pre-fitted input scaler.")

    if label_scaler is None:
        print("Fitting label scaler on training data...")
        label_scaler = MinMaxScaler()
        label_scaler.fit(train_label_files)
    else:
        print("Using pre-fitted label scaler.")

    # --- 3. Create Dataloaders using the fitted or provided scalers ---
    train_batch_iter_numpy, num_train_samples = create_hf_dataset(
        data_dir,
        label_dir,
        split="train",
        val_file_ratio=val_file_ratio,
        shuffle_files_before_split=shuffle_files_before_split,
        seed=seed,
        batch_size=batch_size,
        shuffle_dataset_items=True,
        num_proc=num_proc,
        input_scaler=input_scaler,  # Pass the scaler
        label_scaler=label_scaler,  # Pass the scaler
    )

    def train_jax_batch_iterator():
        for batch in train_batch_iter_numpy():
            yield jnp.array(batch["data"]), jnp.array(batch["label"])

    # Create validation loader
    # Ensure shuffle_files_before_split is True and seed is the same for consistent file splitting
    val_batch_iter_numpy, num_val_samples = create_hf_dataset(
        data_dir,
        label_dir,
        split="validation",
        val_file_ratio=val_file_ratio,
        shuffle_files_before_split=shuffle_files_before_split,
        seed=seed,
        batch_size=batch_size,
        shuffle_dataset_items=False,
        num_proc=num_proc,
        input_scaler=input_scaler,  # Pass the same scaler
        label_scaler=label_scaler,  # Pass the same scaler
    )

    def val_jax_batch_iterator():
        for batch in val_batch_iter_numpy():
            yield jnp.array(batch["data"]), jnp.array(batch["label"])

    return (
        train_jax_batch_iterator,
        num_train_samples,
        val_jax_batch_iterator,
        num_val_samples,
        input_scaler,  # Return the scaler so it can be saved
        label_scaler,  # Return the scaler so it can be saved
    )


# Example usage
def main():
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # dataset_name = "CurveVel_A"  # Example dataset
    dataset_name = "FlatFault_A"  # Example dataset
    # Construct paths relative to the script's location or use absolute paths
    # Assuming OPENFWI is two levels up from the data directory
    base_openfwi_dir = current_dir.parent.parent / "OPENFWI"

    data_dir = base_openfwi_dir / dataset_name / "data"
    label_dir = base_openfwi_dir / dataset_name / "labels"

    if not data_dir.exists() or not label_dir.exists():
        print(f"Data or label directory not found for {dataset_name}:")
        print(f"Data dir: {data_dir}")
        print(f"Label dir: {label_dir}")
        print(
            "Please ensure the OPENFWI dataset is correctly placed or update the paths."
        )
        return

    batch_size = 16
    val_ratio = 0.2  # 20% of files for validation
    random_seed = 42

    print(f"Creating Hugging Face train and validation datasets for {dataset_name}...")
    (
        train_loader,
        num_train_s,
        val_loader,
        num_val_s,
        fitted_input_scaler,  # Capture the scaler
        fitted_label_scaler,  # Capture the scaler
    ) = create_hf_dataset_jax_wrapper(
        str(data_dir),
        str(label_dir),
        val_file_ratio=val_ratio,
        shuffle_files_before_split=True,
        seed=random_seed,
        batch_size=batch_size,
    )

    print(
        f"\nInput Scaler fitted with Mean: {fitted_input_scaler.mean}, Std: {fitted_input_scaler.std}"
    )
    print(
        f"Label Scaler fitted with Min: {fitted_label_scaler.min_val}, Max: {fitted_label_scaler.max_val}"
    )

    print(
        f"\nTraining dataset created. Number of training samples: {num_train_s}. Iterating over a few batches:"
    )
    train_batches_found = False
    for i, (data, label) in enumerate(train_loader()):
        train_batches_found = True
        print(
            f"Train Batch {i+1}: data shape {data.shape}, label shape {label.shape}, data type {data.dtype}"
        )
        if i >= 2:
            break
    if not train_batches_found:
        print("No batches in training loader.")

    print(
        f"\nValidation dataset created. Number of validation samples: {num_val_s}. Iterating over a few batches:"
    )
    val_batches_found = False  # Added for consistency
    for i, (data, label) in enumerate(val_loader()):
        val_batches_found = True  # Added for consistency
        print(
            f"Val Batch {i+1}: data shape {data.shape}, label shape {label.shape}, data type {data.dtype}"
        )
        if i >= 2:
            break
    if not val_batches_found:  # Changed from `i < 0` to use the flag
        print("No batches in validation loader.")


if __name__ == "__main__":
    main()
