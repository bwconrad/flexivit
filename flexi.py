from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from functorch import vmap


def resize(
    x: torch.Tensor,
    shape: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    x_resized = F.interpolate(
        rearrange(x, "h w -> 1 1 h w"), shape, mode=interpolation, antialias=antialias
    )
    return x_resized[0, 0, :, :]


def get_resize_matrix(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
    mat = []
    for i in range(np.prod(old_shape)):
        basis_vec = torch.zeros(old_shape)
        basis_vec[np.unravel_index(i, old_shape)] = 1.0
        mat.append(resize(basis_vec, new_shape).reshape(-1))
    return torch.stack(mat)


def flexi_resample_patch_embed(
    patch_embed: torch.Tensor, new_patch_size: Tuple[int, int]
):
    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_patch_size) == 2, "New patch size should only be [h,w]"

    old_patch_size = tuple(patch_embed.shape[2:])

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    resize_matrix = get_resize_matrix(old_patch_size, new_patch_size)
    resize_matrix_pinv = torch.linalg.pinv(resize_matrix)

    def resample_kernel(kernel: torch.Tensor):
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ kernel.reshape(-1)
        resampled_kernel = rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)
        return resampled_kernel

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    new_kernel = v_resample_kernel(patch_embed)

    return new_kernel


def interpolate_resample_patch_embed(
    patch_embed: torch.Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_patch_size) == 2, "New patch size should only be [h,w]"

    patch_embed = F.interpolate(
        patch_embed, new_patch_size, mode=interpolation, antialias=antialias
    )

    return patch_embed


if __name__ == "__main__":
    from timm.layers.patch_embed import resample_patch_embed

    w = torch.randn([256, 1, 16, 16])
    out1 = flexi_resample_patch_embed(w, (32, 32))
    # out2 = resample_patch_embed(w, [32, 32], interpolation="bilinear", antialias=False)

    out2 = interpolate_resample_patch_embed(w, (32, 32))

    print(out1.size())
    print(out2.size())
    # print(torch.allclose(out1, out2))
