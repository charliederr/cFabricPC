"""
Split-CIFAR Data Loading for Continual Learning.

Provides JAX-compatible data loaders for sequential task learning on
CIFAR-10 and CIFAR-100 split into groups of classes.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Iterator, Sequence, Any
import numpy as np

from fabricpc.utils.data.data_utils import one_hot
from fabricpc.continual.data import TaskData

# CIFAR normalization stats
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

CIFAR100_MEAN = np.array([0.5071, 0.4865, 0.4409], dtype=np.float32)
CIFAR100_STD = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)


class SplitCifarTaskLoader:
    """
    Data loader for a single Split-CIFAR task (a group of classes).

    Yields (images, one_hot_labels). Images are per-channel normalized.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        shuffle: bool = True,
        seed: Optional[int] = None,
        tensor_format: str = "flat",
        num_classes: int = 10,
    ):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.num_classes = num_classes

        self._epoch = 0
        self._num_samples = len(images)
        self._num_batches = (self._num_samples + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(self._num_samples)
        if self.shuffle:
            epoch_seed = self.seed + self._epoch if self.seed is not None else None
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
        self._epoch += 1

        for start in range(0, self._num_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]

            batch_images = self.images[batch_idx].astype(np.float32)
            batch_labels = self.labels[batch_idx]

            # Normalize
            batch_images = (batch_images - self.mean) / self.std

            if self.tensor_format == "flat":
                batch_images = batch_images.reshape(len(batch_idx), -1)
            elif self.tensor_format == "NHWC":
                pass
            else:
                raise ValueError(f"Unknown tensor_format: {self.tensor_format}")

            batch_labels_onehot = one_hot(batch_labels, num_classes=self.num_classes)
            yield batch_images, batch_labels_onehot


def _load_cifar_keras(dataset_name="cifar10"):
    """Load CIFAR via keras.datasets."""
    try:
        from tensorflow.keras.datasets import cifar10, cifar100

        if dataset_name == "cifar10":
            (train_images, train_labels), (test_images, test_labels) = (
                cifar10.load_data()
            )
        else:
            (train_images, train_labels), (test_images, test_labels) = (
                cifar100.load_data(label_mode="fine")
            )
        return train_images, train_labels, test_images, test_labels
    except ImportError:
        return None
    except Exception as e:
        print(
            f"Keras {dataset_name} load failed ({type(e).__name__}: {e}); "
            "falling back to manual download."
        )
        return None


def _load_cifar_manual(dataset_name="cifar10", data_root="./data"):
    """Download CIFAR python tarball and parse it without any TF dependency."""
    import os
    import tarfile
    import pickle
    import shutil
    import urllib.request
    import ssl

    if dataset_name == "cifar10":
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        tar_name = "cifar-10-python.tar.gz"
        extract_name = "cifar-10-batches-py"
    else:
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        tar_name = "cifar-100-python.tar.gz"
        extract_name = "cifar-100-python"

    tar_path = os.path.join(data_root, tar_name)
    extract_dir = os.path.join(data_root, extract_name)
    os.makedirs(data_root, exist_ok=True)

    if not os.path.exists(extract_dir):
        if not os.path.exists(tar_path):
            print(f"Downloading {dataset_name} (~170MB)...")
            try:
                import certifi

                context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                context = ssl.create_default_context()
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=context)
            )
            with opener.open(url) as response, open(tar_path, "wb") as out:
                shutil.copyfileobj(response, out)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_root)

    def _unpickle(path):
        with open(path, "rb") as f:
            return pickle.load(f, encoding="bytes")

    if dataset_name == "cifar10":
        train_images = []
        train_labels = []
        for i in range(1, 6):
            batch = _unpickle(os.path.join(extract_dir, f"data_batch_{i}"))
            train_images.append(batch[b"data"])
            train_labels.extend(batch[b"labels"])
        train_images = (
            np.concatenate(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        )
        train_labels = np.array(train_labels, dtype=np.int32)

        test = _unpickle(os.path.join(extract_dir, "test_batch"))
        test_images = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_labels = np.array(test[b"labels"], dtype=np.int32)
    else:
        train = _unpickle(os.path.join(extract_dir, "train"))
        test = _unpickle(os.path.join(extract_dir, "test"))

        def _reshape(batch):
            data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.array(batch[b"fine_labels"], dtype=np.int32)
            return data, labels

        train_images, train_labels = _reshape(train)
        test_images, test_labels = _reshape(test)

    return train_images, train_labels, test_images, test_labels


class SplitCifarLoader:
    """Base class for Split-CIFAR loaders."""

    def __init__(
        self,
        dataset_name: str,
        class_groups: Optional[Sequence[Sequence[int]]] = None,
        classes_per_task: int = 2,
        num_tasks: int = 5,
        batch_size: int = 256,
        shuffle: bool = True,
        seed: Optional[int] = None,
        tensor_format: str = "flat",
        data_root: str = "./data",
        num_classes: int = 10,
    ):
        if class_groups is None:
            class_groups = tuple(
                tuple(range(i * classes_per_task, (i + 1) * classes_per_task))
                for i in range(num_tasks)
            )
        self.class_groups = [tuple(g) for g in class_groups]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.num_classes = num_classes
        self.dataset_name = dataset_name

        if dataset_name == "cifar10":
            self.mean, self.std = CIFAR10_MEAN, CIFAR10_STD
        else:
            self.mean, self.std = CIFAR100_MEAN, CIFAR100_STD

        self._load_dataset(data_root)

        self.tasks: List[TaskData] = []
        for task_id, classes in enumerate(self.class_groups):
            self.tasks.append(self._create_task(task_id, classes))

    def _load_dataset(self, data_root: str):
        result = _load_cifar_keras(self.dataset_name)
        if result is None:
            result = _load_cifar_manual(self.dataset_name, data_root)
        train_images, train_labels, test_images, test_labels = result

        self._train_images = train_images.astype(np.float32) / 255.0
        self._train_labels = np.asarray(train_labels, dtype=np.int32).reshape(-1)
        self._test_images = test_images.astype(np.float32) / 255.0
        self._test_labels = np.asarray(test_labels, dtype=np.int32).reshape(-1)

    def _create_task(self, task_id: int, classes: Sequence[int]) -> TaskData:
        class_set = set(int(c) for c in classes)

        train_mask = np.isin(self._train_labels, list(class_set))
        train_images = self._train_images[train_mask]
        train_labels = self._train_labels[train_mask]

        test_mask = np.isin(self._test_labels, list(class_set))
        test_images = self._test_images[test_mask]
        test_labels = self._test_labels[test_mask]

        task_seed = self.seed + task_id if self.seed is not None else None

        train_loader = SplitCifarTaskLoader(
            images=train_images,
            labels=train_labels,
            batch_size=self.batch_size,
            mean=self.mean,
            std=self.std,
            shuffle=self.shuffle,
            seed=task_seed,
            tensor_format=self.tensor_format,
            num_classes=self.num_classes,
        )
        test_loader = SplitCifarTaskLoader(
            images=test_images,
            labels=test_labels,
            batch_size=self.batch_size,
            mean=self.mean,
            std=self.std,
            shuffle=False,
            seed=task_seed,
            tensor_format=self.tensor_format,
            num_classes=self.num_classes,
        )

        return TaskData(
            task_id=task_id,
            classes=tuple(classes),
            train_loader=train_loader,
            test_loader=test_loader,
        )

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[TaskData]:
        return iter(self.tasks)

    def __getitem__(self, idx: int) -> TaskData:
        return self.tasks[idx]


class SplitCifar10Loader(SplitCifarLoader):
    def __init__(self, **kwargs):
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = 10
        super().__init__(dataset_name="cifar10", **kwargs)


class SplitCifar100Loader(SplitCifarLoader):
    def __init__(self, **kwargs):
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = 100
        if "classes_per_task" not in kwargs and "class_groups" not in kwargs:
            kwargs["classes_per_task"] = 5
            kwargs["num_tasks"] = 20
        super().__init__(dataset_name="cifar100", **kwargs)


def build_split_cifar10_loaders(config, data_root: str = "./data") -> List[TaskData]:
    loader = SplitCifar10Loader(
        class_groups=getattr(config, "task_pairs", None),
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
        tensor_format=getattr(config.training, "tensor_format", "flat"),
        data_root=data_root,
        num_classes=getattr(config, "num_output_classes", 10),
    )
    return loader.tasks


def build_split_cifar100_loaders(config, data_root: str = "./data") -> List[TaskData]:
    loader = SplitCifar100Loader(
        class_groups=getattr(config, "task_pairs", None),
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
        tensor_format=getattr(config.training, "tensor_format", "flat"),
        data_root=data_root,
        num_classes=getattr(config, "num_output_classes", 100),
    )
    return loader.tasks
