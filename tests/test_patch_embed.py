import numpy as np
import pytest
import torch
import torch.nn.functional as F

from flexivit_pytorch import FlexiPatchEmbed, pi_resize_patch_embed


def _test_patch_emb_resize(old_shape, new_shape, n_patches=100):
    """
    Verifies that if we resize the input image patch and resample
    the patch embedding accordingly, the output does not change.
    NOTE: if the image contains more than one patch, then the embeddings will
    change due to patch interaction during the resizing.
    """

    patch_size = old_shape[2:]
    resized_patch_size = new_shape[2:]

    # Old shape
    patches = torch.randn(n_patches, *old_shape[1:])
    w_emb = torch.randn(*old_shape)
    old_embeddings = F.conv2d(patches, w_emb, stride=patch_size, padding="valid")

    # New shape
    patches_resized = F.interpolate(
        patches, resized_patch_size, mode="bicubic", antialias=True
    )
    w_emb_resampled = pi_resize_patch_embed(
        w_emb, resized_patch_size, interpolation="bicubic", antialias=True
    )
    assert w_emb_resampled.shape == new_shape

    new_embeddings = F.conv2d(
        patches_resized, w_emb_resampled, stride=resized_patch_size, padding="valid"
    )

    assert old_embeddings.shape == new_embeddings.shape

    np.testing.assert_allclose(
        old_embeddings.detach().numpy(),
        new_embeddings.detach().numpy(),
        rtol=1e-1,
        atol=1e-4,
    )


def test_pi_resize_square():
    out_channels = 256
    patch_sizes = [48, 40, 30, 24, 20, 16, 15, 12, 10, 8, 6, 5]
    for s in patch_sizes:
        old_shape = (out_channels, 3, s, s)
        for t in patch_sizes:
            new_shape = (out_channels, 3, t, t)
            if s <= t:
                _test_patch_emb_resize(old_shape, new_shape)


def test_pi_resize_rectangular():
    out_channels = 256
    old_shape = (out_channels, 3, 8, 10)
    new_shape = (out_channels, 3, 10, 12)
    _test_patch_emb_resize(old_shape, new_shape)

    old_shape = (out_channels, 3, 8, 6)
    new_shape = (out_channels, 3, 9, 15)
    _test_patch_emb_resize(old_shape, new_shape)

    old_shape = (out_channels, 3, 8, 6)
    new_shape = (out_channels, 3, 15, 9)
    _test_patch_emb_resize(old_shape, new_shape)


def test_input_channels():
    out_channels = 256
    for c in [1, 3, 10]:
        old_shape = (out_channels, c, 8, 10)
        new_shape = (out_channels, c, 10, 12)
        _test_patch_emb_resize(old_shape, new_shape)


def test_pi_resize_downsampling():
    """
    NOTE: Downsampling does not guarantee that the outputs will match
    before and after. So, the test only checks that the code runs and
    produces an output of the correct shape and type.
    """

    out_channels = 256
    for t in [4, 5, 6, 7]:
        for c in [1, 3, 5]:
            old_shape = (out_channels, c, 8, 8)
            new_shape = (out_channels, c, t, t)

            old = torch.randn(*old_shape)
            resampled = pi_resize_patch_embed(old, new_shape[2:])
            assert resampled.shape == new_shape
            assert resampled.dtype == old.dtype


@pytest.mark.parametrize(
    "old_shape", [(256, 3, 8, 10), (256, 1, 8, 10), (256, 3, 8, 10)]
)
@pytest.mark.parametrize(
    "new_shape", [(256, 1, 10, 12), (256, 3, 10, 12), (256, 3, 10, 12)]
)
def test_pi_resize_incorrect_input_shapes(old_shape, new_shape):
    old = torch.randn(*old_shape)
    with pytest.raises(AssertionError):
        pi_resize_patch_embed(old, new_shape)


def test_flexi_patch_embed_forward():
    for img_size in [224, 240, 256]:
        for patch_size in [32, 16, (41, 27)]:
            for in_chans in [1, 3]:
                pe = FlexiPatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans
                )
                inp = torch.rand(8, in_chans, img_size, img_size)
                y, ps = pe(inp, return_patch_size=True)
                assert y.shape[1] == (img_size // ps[0]) * (img_size // ps[1])


def test_flexi_patch_embed_forward_select_patch_size():
    pe = FlexiPatchEmbed()
    inp = torch.rand(8, 3, 240, 240)
    for ps in [2, 5, 16, 32, 50, 64, (10, 15)]:
        y, ps = pe(inp, patch_size=ps, return_patch_size=True)
        assert y.shape[1] == (240 // ps[0]) * (240 // ps[1])


def test_flexi_patch_embed_forward_eval_uses_base_patch_size():
    for patch_size in [32, 16, (41, 27)]:
        pe = FlexiPatchEmbed(img_size=240, patch_size=patch_size)
        pe.eval()

        inp = torch.rand(8, 3, 240, 240)
        _, ps = pe(inp, return_patch_size=True)

        if isinstance(patch_size, tuple):
            assert patch_size == ps
        else:
            assert (patch_size, patch_size) == ps


def test_flexi_patch_embed_forward_no_patch_size_seq():
    pe = FlexiPatchEmbed(patch_size_seq=[])
    inp = torch.rand(8, 3, 240, 240)

    pe(inp, patch_size=16)
    with pytest.raises(AssertionError):
        pe(inp)


def test_flexi_patch_embed_forward_patch_size_probs():
    pe = FlexiPatchEmbed(patch_size_seq=[16, 32], patch_size_probs=[1.0, 0.0])
    inp = torch.rand(8, 3, 240, 240)

    for _ in range(20):
        _, ps = pe(inp, return_patch_size=True)
        assert ps == (16, 16)
