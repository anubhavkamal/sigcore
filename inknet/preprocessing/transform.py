import numpy as np
from skimage import filters, transform
from typing import Tuple


def prepare_image(img: np.ndarray,
                  canvas_size: Tuple[int, int],
                  img_size: Tuple[int, int] = (170, 242),
                  input_size: Tuple[int, int] = (150, 220)) -> np.ndarray:
    """Pre-process a grayscale signature image: center it, resize, and crop.

    Parameters
    ----------
    img : np.ndarray (H x W)
    canvas_size : tuple (H x W) — canvas to center the signature on
    img_size : tuple (H x W) — intermediate resize target
    input_size : tuple (H x W) — final size after center-crop

    Returns
    -------
    np.ndarray (input_size)
    """
    img = img.astype(np.uint8)
    centered = center_on_canvas(img, canvas_size)
    inverted = 255 - centered
    resized = scale_to_fit(inverted, img_size)

    if input_size is not None and input_size != img_size:
        return center_crop(resized, input_size)
    return resized


def center_on_canvas(img: np.ndarray,
                     canvas_size: Tuple[int, int] = (840, 1360)) -> np.ndarray:
    """Center a signature on a blank canvas and remove background noise via Otsu."""

    blur_radius = 2
    blurred = filters.gaussian(img, blur_radius, preserve_range=True)
    threshold = filters.threshold_otsu(img)

    binarized = blurred > threshold
    r, c = np.where(binarized == 0)
    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())

    cropped = img[r.min(): r.max(), c.min(): c.max()]
    img_rows, img_cols = cropped.shape
    max_rows, max_cols = canvas_size

    r_start = max_rows // 2 - r_center
    c_start = max_cols // 2 - c_center

    if img_rows > max_rows:
        print('Warning: image taller than canvas — cropping height.')
        r_start = 0
        diff = img_rows - max_rows
        cropped = cropped[diff // 2: diff // 2 + max_rows, :]
        img_rows = max_rows
    else:
        extra = (r_start + img_rows) - max_rows
        if extra > 0:
            r_start -= extra
        if r_start < 0:
            r_start = 0

    if img_cols > max_cols:
        print('Warning: image wider than canvas — cropping width.')
        c_start = 0
        diff = img_cols - max_cols
        cropped = cropped[:, diff // 2: diff // 2 + max_cols]
        img_cols = max_cols
    else:
        extra = (c_start + img_cols) - max_cols
        if extra > 0:
            c_start -= extra
        if c_start < 0:
            c_start = 0

    canvas = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    canvas[r_start: r_start + img_rows, c_start: c_start + img_cols] = cropped
    canvas[canvas > threshold] = 255

    return canvas


def strip_background(img: np.ndarray) -> np.ndarray:
    """Remove background noise using Otsu thresholding."""
    img = img.astype(np.uint8)
    threshold = filters.threshold_otsu(img)
    img[img > threshold] = 255
    return img


def scale_to_fit(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image to exactly `size`, preserving aspect ratio then cropping."""
    height, width = size
    w_ratio = float(img.shape[1]) / width
    h_ratio = float(img.shape[0]) / height

    if w_ratio > h_ratio:
        rh, rw = height, int(round(img.shape[1] / h_ratio))
    else:
        rw, rh = width, int(round(img.shape[0] / w_ratio))

    img = transform.resize(img, (rh, rw),
                           mode='constant', anti_aliasing=True, preserve_range=True)
    img = img.astype(np.uint8)

    if w_ratio > h_ratio:
        start = int(round((rw - width) / 2.0))
        return img[:, start: start + width]
    else:
        start = int(round((rh - height) / 2.0))
        return img[start: start + height, :]


def center_crop(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Crop the center of a 2-D image."""
    sy = (img.shape[0] - size[0]) // 2
    sx = (img.shape[1] - size[1]) // 2
    return img[sy: sy + size[0], sx: sx + size[1]]


def center_crop_batch(imgs: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Crop the center of a batch of images (N x C x H x W)."""
    sy = (imgs.shape[2] - size[0]) // 2
    sx = (imgs.shape[3] - size[1]) // 2
    return imgs[:, :, sy: sy + size[0], sx: sx + size[1]]
