import os
from functools import partial
from typing import Callable, Optional, Sequence, Union

import lmdb
import pandas as pd
import pyarrow as pa
import pytorch_lightning as pl
import six
import timm.models
from PIL import Image
from pytorch_lightning.cli import LightningArgumentParser
from timm import create_model
from timm.data import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
                       OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
from timm.data.transforms_factory import transforms_imagenet_eval
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification.accuracy import Accuracy
from torchvision.datasets import ImageFolder

from flexivit_pytorch import (interpolate_resize_patch_embed,
                              pi_resize_patch_embed)
from flexivit_pytorch.utils import resize_abs_pos_embed


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        is_lmdb: bool = False,
        root: str = "data/",
        num_classes: int = 1000,
        size: int = 224,
        crop_pct: float = 1.0,
        interpolation: str = "bicubic",
        mean: Union[Sequence[float], str] = (0.485, 0.456, 0.406),
        std: Union[Sequence[float], str] = (0.229, 0.224, 0.225),
        batch_size: int = 256,
        workers: int = 4,
    ):
        """Classification Evaluation Datamodule

        Args:
            is_lmdb: Whether the dataset is an lmdb file
            root: Path to dataset directory or lmdb file
            num_classes: Number of target classes
            size: Input image size
            crop_pct: Center crop percentage
            mean: Normalization means. Can be 'clip' or 'imagenet' to use the respective defaults
            std: Normalization standard deviations. Can be 'clip' or 'imagenet' to use the respective defaults
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.is_lmdb = is_lmdb
        self.root = root
        self.num_classes = num_classes
        self.size = size
        self.crop_pct = crop_pct
        self.interpolation = interpolation
        self.batch_size = batch_size
        self.workers = workers

        if mean == "clip":
            self.mean = OPENAI_CLIP_MEAN
        elif mean == "imagenet":
            self.mean = IMAGENET_DEFAULT_MEAN
        else:
            self.mean = mean

        if std == "clip":
            self.std = OPENAI_CLIP_STD
        elif std == "imagenet":
            self.std = IMAGENET_DEFAULT_STD
        else:
            self.std = std

        if self.is_lmdb:
            self.dataset_fn = partial(ImageFolderLMDB, db_path=self.root)
            print(f"Using LMDB dataset from {self.root}")
        else:
            self.dataset_fn = partial(ImageFolder, root=self.root)
            print(f"Using dataset in directory {self.root}")

        self.transforms = transforms_imagenet_eval(
            img_size=self.size,
            crop_pct=self.crop_pct,
            interpolation=self.interpolation,
            mean=self.mean,
            std=self.std,
        )

    def setup(self, stage="test"):
        if self.is_lmdb:
            self.test_dataset = ImageFolderLMDB(
                db_path=self.root, transform=self.transforms
            )
            print(f"Using LMDB dataset from {self.root}")
        else:
            self.test_dataset = ImageFolder(root=self.root, transform=self.transforms)
            print(f"Using dataset from {self.root}")

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

        # Load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # Load label
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


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
        self,
        weights: str,
        num_classes: int,
        image_size: int = 224,
        patch_size: int = 16,
        resize_type: str = "pi",
        results_path: Optional[str] = None,
    ):
        """Classification Evaluator

        Args:
            weights: Name of model weights
            n_classes: Number of target class.
            image_size: Size of input images
            patch_size: Resized patch size
            resize_type: Patch embed resize method. One of ["pi", "interpolate"]
            results_path: Path to write evaluation results. Does not write results if empty
        """
        super().__init__()
        self.save_hyperparameters()
        self.weights = weights
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.resize_type = resize_type
        self.results_path = results_path

        # Load original weights
        print(f"Loading weights {self.weights}")
        orig_net = create_model(self.weights, pretrained=True)
        state_dict = orig_net.state_dict()

        # Adjust patch embedding
        if self.resize_type == "pi":
            state_dict["patch_embed.proj.weight"] = pi_resize_patch_embed(
                state_dict["patch_embed.proj.weight"],
                (self.patch_size, self.patch_size),
            )
        elif self.resize_type == "interpolate":
            state_dict["patch_embed.proj.weight"] = interpolate_resize_patch_embed(
                state_dict["patch_embed.proj.weight"],
                (self.patch_size, self.patch_size),
            )
        else:
            raise ValueError(
                f"{self.resize_type} is not a valid value for --model.resize_type. Should be one of ['flexi', 'interpolate']"
            )

        # Adjust position embedding
        if "pos_embed" in state_dict.keys():
            grid_size = self.image_size // self.patch_size
            state_dict["pos_embed"] = resize_abs_pos_embed(
                state_dict["pos_embed"], new_size=(grid_size, grid_size)
            )

        # Load adjusted weights into model with target patch and image sizes
        model_fn = getattr(timm.models, orig_net.default_cfg["architecture"])
        self.net = model_fn(
            img_size=self.image_size,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
        )
        self.net.load_state_dict(state_dict, strict=True)

        # Define metrics
        self.acc = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def test_step(self, batch, _):
        x, y = batch

        # Pass through network
        pred = self(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        acc = self.acc(pred, y)

        # Log
        self.log(f"test_loss", loss)
        self.log(f"test_acc", acc)

        return loss

    def test_epoch_end(self, _):
        if self.results_path:
            acc = self.acc.compute().detach().cpu().item()
            results = pd.DataFrame(
                {
                    "model": [self.weights],
                    "acc": [round(acc, 4)],
                    "patch_size": [self.patch_size],
                    "image_size": [self.image_size],
                    "resize_type": [self.resize_type],
                }
            )

            if not os.path.exists(os.path.dirname(self.results_path)):
                os.makedirs(os.path.dirname(self.results_path))

            results.to_csv(
                self.results_path,
                mode="a",
                header=not os.path.exists(self.results_path),
            )


if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
    parser.add_lightning_class_args(DataModule, "data")
    parser.add_lightning_class_args(ClassificationEvaluator, "model")
    parser.link_arguments("data.num_classes", "model.num_classes")
    parser.link_arguments("data.size", "model.image_size")
    args = parser.parse_args()
    args["logger"] = False  # Disable saving logging artifacts

    dm = DataModule(**args["data"])
    # args["model"]["n_classes"] = dm.num_classes
    # args["model"]["image_size"] = dm.size
    model = ClassificationEvaluator(**args["model"])
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.test(model, datamodule=dm)
