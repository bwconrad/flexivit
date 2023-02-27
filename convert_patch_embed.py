"""
Script to resize a model's patch embedding layer using PI-resizing (Flexivit).
"""

from argparse import ArgumentParser

import torch

from flexivit_pytorch import pi_resize_patch_embed

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Script to resize a model's patch embedding layer using PI-resizing (Flexivit)."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="converted_weights.pt",
        help="Path of output converted weights",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="patch_embed.proj.weight",
        help="Name of patch embedding layer parameter. Should be the weights of a Conv2d layer",
    )
    parser.add_argument(
        "--patch-size",
        "-ps",
        nargs="+",
        required=True,
        help="Output height and width of patch embedding layer",
    )

    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    if "state_dict" in checkpoint.keys():
        new_state_dict = checkpoint["state_dict"]
    else:
        new_state_dict = checkpoint

    print(f"Resizing patch embedding layer from {args.input}")
    print(f"Original patch embedding layer shape: {new_state_dict[args.name].shape}")

    if len(args.patch_size) == 1:
        shape = (int(args.patch_size[0]), int(args.patch_size[0]))
    else:
        shape = (int(args.patch_size[0]), int(args.patch_size[1]))

    new_state_dict[args.name] = pi_resize_patch_embed(new_state_dict[args.name], shape)

    if "state_dict" in checkpoint.keys():
        checkpoint["state_dict"] = new_state_dict
    else:
        checkpoint = new_state_dict

    with open(args.output, "wb") as f:
        torch.save(checkpoint, f)

    print(f"New patch embedding layer shape: {new_state_dict[args.name].shape}")
    print(f"Converted weights saved to {args.output}")
