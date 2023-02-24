import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningArgumentParser
# from timm.layers.patch_embed import \
#     resample_patch_embed as flexi_resample_patch_embed
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torch.nn import CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy

from data import DataModule
from flexi import flexi_resample_patch_embed, interpolate_resample_patch_embed


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
        self,
        weights: str,
        weights_prefix: str = "",
        n_classes: int = 10,
        image_size: int = 224,
        patch_size: int = 16,
        resample_type: str = "flexi",
    ):
        """Classification Evaluator

        Args:
            weights: Path to weights
            weights_prefix: Parameter prefix to strip from weights
            n_classes: Number of target class.
            image_size: Size of input images
            patch_size: Resized patch size
            resample_type: Patch embed resampling method. One of ["flexi", "interpolate"]
        """
        super().__init__()
        self.save_hyperparameters()
        self.weights = weights
        self.weights_prefix = weights_prefix
        self.n_classes = n_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.resample_type = resample_type

        # Initialize model with target patch and image sizes
        self.net = VisionTransformer(
            img_size=self.image_size,
            patch_size=self.patch_size,
            num_classes=self.n_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )

        # Load original weights
        print(f"Loading weights from {self.weights}")
        state_dict = torch.load(self.weights)

        # Remove prefix from key names
        if self.weights_prefix:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(self.weights_prefix):
                    k = k.replace(self.weights_prefix + ".", "")
                    new_state_dict[k] = v
            state_dict = new_state_dict

        # Adjust patch embedding
        if self.resample_type == "flexi":
            patch_embed = flexi_resample_patch_embed(
                state_dict["patch_embed.proj.weight"],
                (self.patch_size, self.patch_size),
            )
        elif self.resample_type == "interpolate":
            patch_embed = interpolate_resample_patch_embed(
                state_dict["patch_embed.proj.weight"],
                (self.patch_size, self.patch_size),
            )
        else:
            raise ValueError(
                f"{self.resample_type} is not a valid value for --model.resample_type. Should be one of ['flex', 'interpolate']"
            )
        state_dict["patch_embed.proj.weight"] = patch_embed

        # Adjust position embedding
        grid_size = self.image_size // self.patch_size
        pos_embed = resample_abs_pos_embed(
            state_dict["pos_embed"], new_size=[grid_size, grid_size]
        )
        state_dict["pos_embed"] = pos_embed

        # Load new weights
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

        # Log
        self.log(f"test_loss", loss)
        self.log(f"test_acc", acc)

        return loss


if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
    parser.add_lightning_class_args(DataModule, "data")
    parser.add_lightning_class_args(ClassificationEvaluator, "model")
    args = parser.parse_args()
    args["logger"] = False  # Disable logging

    dm = DataModule(**args["data"])
    args["model"]["n_classes"] = dm.num_classes
    args["model"]["image_size"] = dm.crop
    model = ClassificationEvaluator(**args["model"])
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.test(model, datamodule=dm)
