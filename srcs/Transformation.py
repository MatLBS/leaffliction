import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


def display_images(images: list):
    cols = 4
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, (image, title) in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def corner_detecttion(img: np.ndarray) -> np.ndarray:
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect corners using the Harris method
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Create a boolean bitmap of corner positions
    corners = dst > 0.03 * dst.max()

    # Find the coordinates from the boolean bitmap
    coord = np.argwhere(corners)

    # Draw circles on the coordinates to mark the corners
    for y, x in coord:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    return img


def canny_edge(img: np.ndarray) -> np.ndarray:
    img = img.copy()
    edges = cv2.Canny(img, 100, 200)

    return edges


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    img = img.copy()
    Gaussian_rgb = cv2.GaussianBlur(img, (5, 5), 0)

    return Gaussian_rgb


def leaf_mask(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary


def mask(img: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    img = img.copy()
    masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
    masked_img[binary_mask == 0] = (255, 255, 255)

    return masked_img


def LBP(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="default")

    return lbp.astype(np.uint8)


def ROI(img: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    img = img.copy()

    green_layer = np.zeros_like(img)
    green_layer[:, :] = (0, 255, 0)

    alpha = 0.3
    blended = cv2.addWeighted(img, alpha, green_layer, 1 - alpha, 0)

    final_image = img.copy()
    final_image[binary_mask == 255] = blended[binary_mask == 255]

    h, w = final_image.shape[:2]
    cv2.rectangle(final_image, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    return final_image


def transform_image(image_path: str, show_images: bool = True) -> list:
    img = cv2.imread(image_path)
    binary_mask = leaf_mask(img)

    cornered_image = corner_detecttion(img)
    blurred_image = gaussian_blur(img)
    edged_image = canny_edge(img)
    lbp_image = LBP(img)
    masked_image = mask(img, binary_mask)
    roi_image = ROI(img, binary_mask)

    transformed_images = [
        (img, "Original"),
        (cornered_image, "Harris_corners"),
        (blurred_image, "Gaussian_blurred"),
        (edged_image, "Edges"),
        (masked_image, "Masked"),
        (lbp_image, "LBP"),
        (roi_image, "ROI"),
    ]

    if show_images:
        display_images(transformed_images)
        return

    return transformed_images


def transform_all_images(src_dir: str, dst_dir: str) -> None:
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        if os.path.isfile(src_path):
            transfomed_images = transform_image(src_path, show_images=False)
            for _, (image, transformation) in enumerate(transfomed_images):
                output_path = os.path.splitext(dst_path)[0]
                if transformation == "Original":
                    continue
                cv2.imwrite(
                    f"{output_path}_{transformation}.jpg",
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
        elif os.path.isdir(src_path):
            transform_all_images(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Image transformation")
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        default=None,
        help="Path to a single image to transform",
    )
    parser.add_argument(
        "-src",
        type=str,
        default=None,
        metavar="N",
        help="Source directory path",
    )
    parser.add_argument(
        "-dst",
        type=str,
        default=None,
        metavar="N",
        help="Destination directory path",
    )
    args = parser.parse_args()
    try:
        if args.src and args.dst:
            transform_all_images(args.src, args.dst)
        elif args.image:
            assert os.path.exists(args.image), "The path does not exist"
            assert os.path.isfile(args.image), "The path must be a file"
            transform_image(args.image, True)
        else:
            parser.error("You must provide an image path or -src and -dst")
    except (ValueError, AssertionError) as error:
        print(type(error).__name__ + ":", error)


if __name__ == "__main__":
    main()
