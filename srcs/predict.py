import argparse
import os
import torch
from cnn import CNN
import torchvision.transforms as transforms
from PIL import Image


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


def make_prediction(model_path: str, image_path: str) -> None:
    model = CNN()

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["state_dict"])

    classes = state_dict["classes"]
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    tensor = transform(image)

    prediction = model.predict(tensor, classes)
    print("Prediction:", prediction)


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
