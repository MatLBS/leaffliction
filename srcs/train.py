import argparse
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from cnn import CNN


def get_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Play Snake")
    parser.add_argument(
        "-src",
        type=str,
        default=None,
        metavar="N",
        help="Apply the transformation to all images in the dataset (provide the folder path)",
    )
    return parser.parse_args(), parser


def check_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.src:
        assert os.path.exists(args.src), "The path does not exist"
        assert os.path.isdir(args.src), "The path must be a directory"
    else:
        parser.error("You must specify the source folder path")


def load_data(src: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=src, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return dataset, train_loader, test_loader


def train_model(dataset_path: str) -> None:
    dataset, train_loader, test_loader = load_data(dataset_path, 256)
    cnn = CNN(dataset=dataset)

    cnn.fit(train_loader, test_loader)
    cnn.test(test_loader)
    cnn.save_model()
    cnn.display_loss()


def main():
    args, parser = get_args()
    try:
        check_args(args, parser)
        if args.src:
            train_model(args.src)
        else:
            parser.error("You must specify the source folder path")
    except (ValueError, AssertionError) as error:
        print(type(error).__name__ + ":", error)


if __name__ == "__main__":
    main()
