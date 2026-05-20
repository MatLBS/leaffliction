import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

from Distribution import count_files

IMG_SIZE = 180


def shear_image(image, shear_x=0.0, shear_y=0.0):
    h, w = image.shape[:2]

    M = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def save_augemented_images(images: list, output_path: str):
    output_path = os.path.splitext(output_path)[0]
    for i, (image, augmentation) in enumerate(images):
        if augmentation == "Original":
            continue
        cv2.imwrite(
            f"{output_path}_{augmentation}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )


def transform_single_image(image_path: str, show_images: bool = True) -> None:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    flipped = cv2.flip(img, 1)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    blur = cv2.blur(img, (10, 10))
    contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    cropped = img[20 : 20 + IMG_SIZE, 20 : 20 + IMG_SIZE]
    sheared = shear_image(img, shear_x=0.2, shear_y=0.2)

    images = [
        (img, "Original"),
        (flipped, "Flipped"),
        (rotated, "Rotated"),
        (blur, "Blurred"),
        (contrast, "Contrast"),
        (cropped, "Cropped"),
        (sheared, "Sheared"),
    ]

    if show_images:
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

    save_augemented_images(images, image_path)


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

    for name, count in counts.items():
        nb_images_to_augment = max_count - count
        image_paths = get_image_paths(folder_path, name)
        it = iter(image_paths)
        while nb_images_to_augment > 0:
            transform_single_image(next(it), show_images=False)
            nb_images_to_augment -= 6

    create_augmented_directory(folder_path)


def main():
    parser = argparse.ArgumentParser(description="Play Snake")
    parser.add_argument(
        "--all",
        type=str,
        default=None,
        metavar="N",
        help="Apply the transformation to all images in the dataset (provide the folder path)",
    )
    parser.add_argument(
        "--specific",
        type=str,
        default=None,
        metavar="N",
        help="Apply the transformation to a specific image (provide the file path)",
    )
    args = parser.parse_args()
    try:
        if args.all:
            transform_all_images(args.all)
        elif args.specific:
            transform_single_image(args.specific, True)
        else:
            raise ValueError("You must specify either --all or --specific")
    except ValueError as error:
        print(ValueError.__name__ + ":", error)


if __name__ == "__main__":
    main()
