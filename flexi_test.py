import numpy as np
import torch
import torch.nn.functional as F
from absl.testing import absltest
from timm.layers.patch_embed import \
    resample_patch_embed as flexi_resample_patch_embed

# from flexi import flexi_resample_patch_embed


class PatchEmbedTest(absltest.TestCase):
    def _test_patch_emb_resize(self, old_shape, new_shape, n_patches=100):
        # This test verifies that if we resize the input image patch and resample
        # the patch embedding accordingly, the output does not change.
        # NOTE: if the image contains more than one patch, then the embeddings will
        # change due to patch interaction during the resizing.

        patch_size = old_shape[2:]
        resized_patch_size = new_shape[2:]

        # Old shape
        patches = torch.randn(n_patches, *old_shape[1:])
        w_emb = torch.randn(*old_shape)
        old_embeddings = F.conv2d(patches, w_emb, stride=patch_size, padding="valid")

        # New shape
        patches_resized = F.interpolate(patches, resized_patch_size, mode="bilinear")
        w_emb_resampled = flexi_resample_patch_embed(
            w_emb, resized_patch_size, interpolation="bilinear", antialias=True
        )
        self.assertEqual(w_emb_resampled.shape, new_shape)

        new_embeddings = F.conv2d(
            patches_resized, w_emb_resampled, stride=resized_patch_size, padding="valid"
        )

        self.assertEqual(old_embeddings.shape, new_embeddings.shape)
        np.testing.assert_allclose(
            old_embeddings.detach().numpy(),
            new_embeddings.detach().numpy(),
            rtol=1e-1,
            atol=1e-4,
        )

    def test_resize_square(self):
        out_channels = 256
        patch_sizes = [48, 40, 30, 24, 20, 16, 15, 12, 10, 8, 6, 5]
        for s in patch_sizes:
            old_shape = (out_channels, 3, s, s)
            for t in patch_sizes:
                new_shape = (out_channels, 3, t, t)
                if s <= t:
                    self._test_patch_emb_resize(old_shape, new_shape)

    def test_resize_rectangular(self):
        out_channels = 256
        old_shape = (out_channels, 3, 8, 10)
        new_shape = (out_channels, 3, 10, 12)
        self._test_patch_emb_resize(old_shape, new_shape)

        old_shape = (out_channels, 3, 8, 6)
        new_shape = (out_channels, 3, 9, 15)
        self._test_patch_emb_resize(old_shape, new_shape)

        old_shape = (out_channels, 3, 8, 6)
        new_shape = (out_channels, 3, 15, 9)
        self._test_patch_emb_resize(old_shape, new_shape)

    def test_input_channels(self):
        out_channels = 256
        for c in [1, 3, 10]:
            old_shape = (out_channels, c, 8, 10)
            new_shape = (out_channels, c, 10, 12)
            self._test_patch_emb_resize(old_shape, new_shape)

    def _test_works(self, old_shape, new_shape):
        old = torch.randn(*old_shape)
        resampled = flexi_resample_patch_embed(old, new_shape[2:])
        self.assertEqual(resampled.shape, new_shape)
        self.assertEqual(resampled.dtype, old.dtype)

    def test_downsampling(self):
        # NOTE: for downsampling we cannot guarantee that the outputs would match
        # before and after downsampling. So, we simply test that the code runs and
        # produces an output of the correct shape and type.
        out_channels = 256
        for t in [4, 5, 6, 7]:
            for c in [1, 3, 5]:
                old_shape = (out_channels, c, 8, 8)
                new_shape = (out_channels, c, t, t)
                self._test_works(old_shape, new_shape)

    def _test_raises(self, old_shape, new_shape):
        old = torch.randn(*old_shape)
        with self.assertRaises(AssertionError):
            flexi_resample_patch_embed(old, new_shape)

    def test_raises_incorrect_dims(self):
        old_shape = (256, 3, 8, 10)
        new_shape = (256, 1, 10, 12)
        self._test_raises(old_shape, new_shape)

        old_shape = (256, 1, 8, 10)
        new_shape = (256, 3, 10, 12)
        self._test_raises(old_shape, new_shape)

        old_shape = (256, 3, 8, 10)
        new_shape = (256, 3, 10, 12)
        self._test_raises(old_shape, new_shape)


if __name__ == "__main__":
    absltest.main()
