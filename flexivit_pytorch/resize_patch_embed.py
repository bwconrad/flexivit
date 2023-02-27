from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from functorch import vmap


def pi_resize_patch_embed(
    patch_embed: torch.Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4d tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (h, w)"

    old_patch_size = tuple(patch_embed.shape[2:])

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: torch.Tensor, shape: Tuple[int, int]):
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=interpolation,
            antialias=antialias,
        )
        return x_resized[0, 0, ...]

    def calculate_resize_pinv(old_shape: Tuple[int, int], new_shape: Tuple[int, int]):
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_resize_pinv(old_patch_size, new_patch_size)

    def resample_patch_embed(patch_embed: torch.Tensor):
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)


def interpolate_resize_patch_embed(
    patch_embed: torch.Tensor,
    new_patch_size: Tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4d tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (h, w)"

    patch_embed = F.interpolate(
        patch_embed, new_patch_size, mode=interpolation, antialias=antialias
    )

    return patch_embed


if __name__ == "__main__":
    w = torch.randn([256, 1, 16, 16])
    out1 = pi_resize_patch_embed(w, (32, 32))
    out2 = interpolate_resize_patch_embed(w, (32, 32))

    print(out1.size())
    print(out2.size())
