import os
from functools import partial
from typing import Callable, Optional, Sequence

import lmdb
import pyarrow as pa
import pytorch_lightning as pl
import six
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import (CIFAR10, CIFAR100, DTD, STL10, FGVCAircraft,
                                  Flowers102, Food101, ImageFolder,
                                  OxfordIIITPet, StanfordCars)
from torchvision.transforms import InterpolationMode

DATASET_DICT = {
    "cifar10": [partial(CIFAR10, train=False, download=True), 10],
    "cifar100": [partial(CIFAR100, train=False, download=True), 100],
    "flowers102": [partial(Flowers102, split="test", download=True), 102],
    "food101": [partial(Food101, split="test", download=True), 101],
    "pets37": [partial(OxfordIIITPet, split="test", download=True), 37],
    "stl10": [partial(STL10, split="test", download=True), 10],
    "dtd": [partial(DTD, split="test", download=True), 47],
    "aircraft": [partial(FGVCAircraft, split="test", download=True), 100],
    "cars": [partial(StanfordCars, split="test", download=True), 196],
}


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "cifar10",
        root: str = "data/",
        num_classes: Optional[int] = None,
        size: int = 256,
        crop: int = 224,
        mean: Sequence = (0.485, 0.456, 0.406),
        std: Sequence = (0.229, 0.224, 0.225),
        batch_size: int = 256,
        workers: int = 4,
    ):
        """Classification Datamodule

        Args:
            dataset: Name of dataset. One of [custom, cifar10, cifar100, flowers102
                     food101, pets37, stl10, dtd, aircraft, cars]
            root: Download path for built-in datasets or path to dataset directory for custom datasets
            num_classes: Number of classes when using a custom dataset
            size: Size after resize
            crop: Size of center crop
            mean: Normalization means
            std: Normalization standard deviations
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.root = root
        self.size = size
        self.crop = crop
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.workers = workers

        # Define dataset
        if self.dataset == "custom":
            assert (
                num_classes is not None
            ), "Must set --data.num_classes when using a custom dataset"
            self.num_classes = num_classes

            self.dataset_fn = partial(ImageFolder, root=self.root)
            print(f"Using custom dataset from {self.root}")
        elif self.dataset == "lmdb":
            assert (
                num_classes is not None
            ), "Must set --data.num_classes when using an lmdb dataset"
            self.num_classes = num_classes

            self.dataset_fn = partial(ImageFolderLMDB, db_path=self.root)

        else:
            try:
                self.dataset_fn, self.num_classes = DATASET_DICT[self.dataset]
                print(f"Using the {self.dataset} dataset")
            except:
                raise ValueError(
                    f"{dataset} is not an available dataset. Should be one of {[k for k in DATASET_DICT.keys()]}"
                )

        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.size, self.size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop((self.crop, self.crop)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def prepare_data(self):
        if self.dataset not in ["custom", "lmdb"]:
            self.dataset_fn(self.root)

    def setup(self, stage="test"):
        if self.dataset in ["custom", "lmdb"]:
            if stage == "test":
                self.test_dataset = self.dataset_fn(transform=self.transforms)
        else:
            if stage == "test":
                self.test_dataset = self.dataset_fn(
                    self.root, transform=self.transforms, download=False
                )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )


class ImageFolderLMDB(Dataset):
    def __init__(
        self,
        db_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=os.path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b"__len__"))
            self.keys = pa.deserialize(txn.get(b"__keys__"))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"


if __name__ == "__main__":
    d = ImageFolderLMDB("data/val.lmdb")
    print(len(d))
