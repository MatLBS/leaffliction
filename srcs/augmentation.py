import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from distribution import count_files

IMG_SIZE = 180


def get_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Image augmentation")
    parser.add_argument(
        "-src",
        type=str,
        default=None,
        metavar="N",
        help="Apply the transformation to all images in a directory (provide the directory path)",
    )
    parser.add_argument(
        "--specific",
        type=str,
        default=None,
        metavar="N",
        help="Apply the transformation to a specific image (provide the file path)",
    )
    return parser.parse_args(), parser


def check_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.src:
        assert os.path.exists(args.src), "The path does not exist"
        assert os.path.isdir(args.src), "The path must be a directory"
    elif args.specific:
        assert os.path.exists(args.specific), "The path does not exist"
        assert os.path.isfile(args.specific), "The path must be a file"
    else:
        parser.error("You must specify either -src or --specific")


def shear_image(image: np.ndarray, shear_x: float, shear_y: float) -> np.ndarray:
    h, w = image.shape[:2]

    M = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def save_augemented_images(images: list, output_path: str) -> None:
    rel = output_path.split(os.sep, 1)[1]
    rel = os.path.splitext(rel)[0]
    dest = os.path.join("augmented_directory", rel)

    for image, augmentation in images:
        if augmentation == "Original":
            continue
        cv2.imwrite(
            f"{dest}_{augmentation}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )


def display_images(images: list) -> None:
    cols = 4
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, (image, title) in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def augment_image(image_path: str, show_images: bool = True) -> None:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    flipped = cv2.flip(img, 1)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    blur = cv2.blur(img, (10, 10))
    contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    cropped = img[20 : 20 + IMG_SIZE, 20 : 20 + IMG_SIZE]
    sheared = shear_image(img, 0.2, 0.2)

    augmented_images = [
        (img, "Original"),
        (flipped, "Flipped"),
        (rotated, "Rotated"),
        (blur, "Blurred"),
        (contrast, "Contrast"),
        (cropped, "Cropped"),
        (sheared, "Sheared"),
    ]

    if show_images:
        display_images(augmented_images)
        return

    save_augemented_images(augmented_images, image_path)


def get_image_paths(folder_path: str, name: str) -> list:
    class_folder = os.path.join(folder_path, name)
    image_files = [
        os.path.join(class_folder, f)
        for f in os.listdir(class_folder)
        if os.path.isfile(os.path.join(class_folder, f))
    ]
    return image_files


def create_augmented_directory(folder_path: str) -> None:
    augmented_dir = os.path.join(os.getcwd(), "augmented_directory")
    if os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)
    shutil.copytree(folder_path, augmented_dir)


def transform_all_images(folder_path: str) -> None:
    counts = count_files(folder_path)
    max_count = max(counts.values()) if counts else 0

    create_augmented_directory(folder_path)

    for name, count in counts.items():
        nb_images_to_augment = max_count - count
        image_paths = get_image_paths(folder_path, name)
        it = iter(image_paths)
        while nb_images_to_augment > 0:
            augment_image(next(it), show_images=False)
            nb_images_to_augment -= 6


def main():
    args, parser = get_args()
    try:
        check_args(args, parser)
        if args.src:
            transform_all_images(args.src)
        elif args.specific:
            augment_image(args.specific, True)
    except (ValueError, AssertionError) as error:
        print(type(error).__name__ + ":", error)


if __name__ == "__main__":
    main()
