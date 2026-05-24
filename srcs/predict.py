import argparse
import os
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from cnn import CNN
import torchvision.transforms as transforms
from PIL import Image
from transformation import (
    apply_specific_transformation,
)


def get_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Play Snake")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="N",
        help="Path to the model to use for prediction (provide the file path)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        metavar="N",
        help="Path to the image to use for prediction (provide the file path)",
    )
    return parser.parse_args(), parser


def check_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.model:
        assert os.path.exists(args.model), "The path does not exist"
        assert os.path.isfile(args.model), "The path must be a file"
    else:
        parser.error("You must specify the model file path")
    if args.image:
        assert os.path.exists(args.image), "The path does not exist"
        assert os.path.isfile(args.image), "The path must be a file"
    else:
        parser.error("You must specify the image file path")


def to_pil_rgb(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img).convert("RGB")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def display_prediction(
    original_bgr: np.ndarray,
    transformed_img: Image.Image,
    prediction: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].set_title("Transformed")
    axes[1].imshow(transformed_img)
    axes[1].axis("off")
    fig.suptitle(f"Prediction: {prediction}", fontsize=14)
    plt.tight_layout()
    plt.show()


def make_prediction(model_path: str, image_path: str) -> None:
    model = CNN()

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["state_dict"])
    classes = state_dict["classes"]

    original_img = cv2.imread(image_path)
    transformed_img = apply_specific_transformation(image_path)

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    tensor = transform(to_pil_rgb(transformed_img))

    prediction = model.predict(tensor, classes)
    display_prediction(original_img, transformed_img, prediction)


def main():
    args, parser = get_args()
    try:
        check_args(args, parser)
        if args.model and args.image:
            make_prediction(args.model, args.image)
        else:
            parser.error("You must specify both the model and image file paths")
    except (ValueError, AssertionError) as error:
        print(type(error).__name__ + ":", error)


if __name__ == "__main__":
    main()
