import os

import pandas as pd
import pytorch_lightning as pl
import timm.models
from pytorch_lightning.cli import LightningArgumentParser
from timm import create_model
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.nn import CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy

from data import DataModule
from flexi import flexi_resample_patch_embed, interpolate_resample_patch_embed


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
        self,
        weights: str,
        n_classes: int = 10,
        image_size: int = 224,
        patch_size: int = 16,
        resample_type: str = "flexi",
        results_path: str = "",
    ):
        """Classification Evaluator

        Args:
            weights: Name of model weights
            n_classes: Number of target class.
            image_size: Size of input images
            patch_size: Resized patch size
            resample_type: Patch embed resampling method. One of ["flexi", "interpolate"]
            results_path: Path to write evaluation results. Does not write results if empty
        """
        super().__init__()
        self.save_hyperparameters()
        self.weights = weights
        self.n_classes = n_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.resample_type = resample_type
        self.results_path = results_path

        # Load original weights
        print(f"Loading weights {self.weights}")
        orig_net = create_model(self.weights, pretrained=True)
        state_dict = orig_net.state_dict()

        # Adjust patch embedding
        if self.resample_type == "flexi":
            state_dict["patch_embed.proj.weight"] = flexi_resample_patch_embed(
                state_dict["patch_embed.proj.weight"],
                (self.patch_size, self.patch_size),
            )
        elif self.resample_type == "interpolate":
            state_dict["patch_embed.proj.weight"] = interpolate_resample_patch_embed(
                state_dict["patch_embed.proj.weight"],
                (self.patch_size, self.patch_size),
            )
        else:
            raise ValueError(
                f"{self.resample_type} is not a valid value for --model.resample_type. Should be one of ['flexi', 'interpolate']"
            )

        # Adjust position embedding
        grid_size = self.image_size // self.patch_size
        state_dict["pos_embed"] = resample_abs_pos_embed(
            state_dict["pos_embed"], new_size=[grid_size, grid_size]
        )

        # Load adjusted weights into model with target patch and image sizes
        model_fn = getattr(timm.models, orig_net.default_cfg["architecture"])
        self.net = model_fn(
            img_size=self.image_size,
            patch_size=self.patch_size,
            num_classes=self.n_classes,
        )
        self.net.load_state_dict(state_dict, strict=True)

        # Define metrics
        self.acc = Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1)

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

        return loss

    def test_epoch_end(self, _) -> None:
        if self.results_path:
            acc = self.acc.compute().detach().cpu().item()
            results = pd.DataFrame(
                {
                    "model": [self.weights],
                    "acc": [round(acc, 4)],
                    "patch_size": [self.patch_size],
                    "image_size": [self.image_size],
                    "resample_type": [self.resample_type],
                }
            )

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
    args = parser.parse_args()
    args["logger"] = False  # Disable logging

    dm = DataModule(**args["data"])
    args["model"]["n_classes"] = dm.num_classes
    args["model"]["image_size"] = dm.size
    model = ClassificationEvaluator(**args["model"])
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.test(model, datamodule=dm)
