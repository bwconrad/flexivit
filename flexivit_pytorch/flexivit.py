from functools import partial
from typing import Optional, Sequence, Tuple, Union

import torch
from patch_embed import FlexiPatchEmbed
from timm.models.vision_transformer import Block, VisionTransformer
from torch import Tensor, nn
from utils import resize_abs_pos_embed, to_2tuple


class FlexiViT(VisionTransformer):
    def __init__(
        self,
        img_size: int = 240,
        base_patch_size: Union[int, Tuple[int, int]] = 32,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = True,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        weight_init: str = "",
        embed_layer: nn.Module = FlexiPatchEmbed,  # type:ignore
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
        block_fn: nn.Module = Block,  # type:ignore
        patch_size_seq: Sequence[int] = (8, 10, 12, 15, 16, 20, 14, 30, 40, 48),
        base_pos_embed_size: Union[int, Tuple[int, int]] = 7,
        patch_size_probs: Optional[Sequence[float]] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """Vision transformer w/ flexible patch sizes

        From: https://arxiv.org/abs/2212.08013

        Args:
            img_size: input image size
            patch_size: patch size
            in_chans: number of input channels
            num_classes: number of classes for classification head
            global_pool: type of global pooling for final sequence (default: 'token')
            embed_dim: embedding dimension
            depth: depth of transformer
            num_heads: number of attention heads
            mlp_ratio: ratio of mlp hidden dim to embedding dim
            qkv_bias: enable bias for qkv if True
            init_values: layer-scale init values
            class_token: use class token
            fc_norm: pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate: dropout rate
            attn_drop_rate: attention dropout rate
            drop_path_rate: stochastic depth rate
            weight_init: weight init scheme
            embed_layer: patch embedding layer
            norm_layer: normalization layer
            act_layer: MLP activation layer
            patch_size_seq: List of patch sizes to randomly sample from
            base_pos_embed_size: Base position embedding size. i.e. the size of the parameter buffer
            patch_size_probs: Optional list of probabilities of sample corresponding
                patch_size_seq element. If None, then uniform distribution is used
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """

        assert embed_layer == FlexiPatchEmbed, "embed_layer should be a FlexiPatchEmbed"

        # Pre-initialize the flexi specific arguments
        embed_layer_fn = partial(
            FlexiPatchEmbed,
            patch_size_seq=patch_size_seq,
            patch_size_probs=patch_size_probs,
            grid_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
        )

        # Position embedding resizing function
        self.resize_pos_embed = partial(
            resize_abs_pos_embed,
            old_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
            num_prefix_tokens=1 if class_token and not no_embed_class else 0,
        )

        self.img_size = to_2tuple(img_size)

        super().__init__(
            img_size,
            base_patch_size,  # type:ignore
            in_chans,
            num_classes,
            global_pool,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            init_values,
            class_token,
            no_embed_class,
            pre_norm,
            fc_norm,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            weight_init,
            embed_layer_fn,  # type:ignore
            norm_layer,
            act_layer,
            block_fn,  # type:ignore
        )

    def _pos_embed(self, x: Tensor, patch_size: Tuple[int, int]) -> Tensor:
        # Resize position embedding based on current patch size
        new_size = (
            int(self.img_size[0] // patch_size[0]),
            int(self.img_size[1] // patch_size[1]),
        )
        pos_embed = self.resize_pos_embed(self.pos_embed, new_size)

        if self.no_embed_class:
            # Position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # Position embedding has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward_features(
        self, x: Tensor, patch_size: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Tensor:
        x, ps = self.patch_embed(x, patch_size, return_patch_size=True)
        x = self._pos_embed(x, ps)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(
        self, x: Tensor, patch_size: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Tensor:
        x = self.forward_features(x, patch_size)
        x = self.forward_head(x)
        return x


def flexivit_tiny(**kwargs) -> FlexiViT:
    return FlexiViT(embed_dim=192, depth=12, num_heads=3, **kwargs)


def flexivit_small(**kwargs) -> FlexiViT:
    return FlexiViT(embed_dim=384, depth=12, num_heads=6, **kwargs)


def flexivit_base(**kwargs) -> FlexiViT:
    return FlexiViT(embed_dim=768, depth=12, num_heads=12, **kwargs)


def flexivit_large(**kwargs) -> FlexiViT:
    return FlexiViT(embed_dim=1024, depth=24, num_heads=16, **kwargs)


def flexivit_huge(**kwargs) -> FlexiViT:
    return FlexiViT(embed_dim=1280, depth=32, num_heads=16, **kwargs)


if __name__ == "__main__":
    x = torch.rand(2, 3, 224, 224)
    x = x.cuda()
    f = flexivit_tiny(img_size=224)
    f = f.cuda()
    import time

    # from timm import create_model
    # f = create_model("vit_tiny_patch16_224", img_size=240, patch_size=16).cuda()

    s = time.time()
    for i in range(100):
        f(x)

    print(time.time() - s)
