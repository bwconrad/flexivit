import pytest
import torch

from flexivit_pytorch.flexivit import (flexivit_base, flexivit_huge,
                                       flexivit_large, flexivit_small,
                                       flexivit_tiny)


@pytest.mark.parametrize(
    "model",
    [flexivit_tiny, flexivit_small, flexivit_base, flexivit_large, flexivit_huge],
)
def test_flexivit_forward(model):
    for image_size in [224, 240]:
        for in_chans in [1, 3]:
            for pos_embed_size in [7, 10, (4, 3)]:
                net = model(
                    img_size=image_size,
                    in_chans=in_chans,
                    base_pos_embed_size=pos_embed_size,
                    num_classes=10,
                )
                net.eval()

                inp = torch.randn(1, in_chans, image_size, image_size)
                out = net(inp)

                assert out.shape[0] == 1
                assert not torch.isnan(out).any(), "Output included NaNs"


@pytest.mark.parametrize(
    "model",
    [flexivit_tiny, flexivit_small, flexivit_base, flexivit_large, flexivit_huge],
)
def test_flexivit_backward(model):
    net = model(img_size=240, num_classes=10)
    num_params = sum([x.numel() for x in net.parameters()])
    net.train()

    inp = torch.randn(1, 3, 240, 240)
    out = net(inp)

    out.mean().backward()
    for n, x in net.named_parameters():
        assert x.grad is not None, f"No gradient for {n}"
    num_grad = sum([x.grad.numel() for x in net.parameters() if x.grad is not None])

    assert out.shape[-1] == 10
    assert num_params == num_grad, "Some parameters are missing gradients"
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_flexivit_no_embed_class():
    net = flexivit_tiny(no_embed_class=True)
    inp = torch.randn(1, 3, 240, 240)
    net(inp)

    net = flexivit_tiny(no_embed_class=False)
    inp = torch.randn(1, 3, 240, 240)
    net(inp)


def test_flexivit_forward_select_patch_size():
    net = flexivit_tiny()
    inp = torch.randn(1, 3, 240, 240)
    net(inp, patch_size=16)
    net(inp, patch_size=20)

    y = net.forward_features(inp, patch_size=20)
    assert y.shape[1] == (240 // 20) ** 2 + 1

    y = net.forward_features(inp, patch_size=(25, 35))
    assert y.shape[1] == (240 // 25) * (240 // 35) + 1
