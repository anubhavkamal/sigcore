import numpy as np
import functools
from tqdm import tqdm
from inknet.datasets.base import SignatureSource
from inknet.preprocessing.transform import prepare_image
from typing import Tuple, Callable, Dict, Union


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray]:
    """Load a pre-processed dataset stored as a .npz file.

    The file must contain arrays: x, y, yforg, usermapping, filenames.
    """
    with np.load(path, allow_pickle=True) as data:
        x = data['x']
        y = data['y']
        yforg = data['yforg']
        user_mapping = data['user_mapping']
        filenames = data['filenames']
    return x, y, yforg, user_mapping, filenames


def build_dataset(source: SignatureSource,
                  save_path: str,
                  img_size: Tuple[int, int],
                  subset: slice = slice(None)):
    """Process a SignatureSource, normalize images, and save as .npz."""
    preprocess_fn = functools.partial(
        prepare_image,
        canvas_size=source.maxsize,
        img_size=img_size,
        input_size=img_size,
    )
    x, y, yforg, user_mapping, used_files = _process_images(source, preprocess_fn, img_size, subset)
    np.savez(save_path, x=x, y=y, yforg=yforg,
             user_mapping=user_mapping, filenames=used_files)
    return x, y, yforg, user_mapping, used_files


def _process_images(source: SignatureSource,
                    preprocess_fn: Callable[[np.ndarray], np.ndarray],
                    img_size: Tuple[int, int],
                    subset: slice) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    user_mapping: Dict[int, int] = {}
    users = source.get_user_list()[subset]
    print('Users to process: %d' % len(users))

    H, W = img_size
    max_sigs = len(users) * (source.genuine_per_user + source.skilled_per_user + source.simple_per_user)

    x = np.empty((max_sigs, H, W), dtype=np.uint8)
    y = np.empty(max_sigs, dtype=np.int32)
    yforg = np.empty(max_sigs, dtype=np.int32)
    used_files = []

    N = 0
    for i, user in enumerate(tqdm(users)):
        gen_data = [(preprocess_fn(img), fname) for img, fname in source.iter_genuine(user)]
        gen_imgs, gen_fnames = zip(*gen_data)
        cnt = len(gen_imgs)
        x[N: N + cnt] = gen_imgs
        yforg[N: N + cnt] = 0
        y[N: N + cnt] = i
        used_files += list(gen_fnames)
        N += cnt

        forg_data = [(preprocess_fn(img), fname) for img, fname in source.iter_forgery(user)]
        if forg_data:
            forg_imgs, forg_fnames = zip(*forg_data)
            cnt = len(forg_imgs)
            x[N: N + cnt] = forg_imgs
            yforg[N: N + cnt] = 1
            y[N: N + cnt] = i
            used_files += list(forg_fnames)
            N += cnt

        simple_data = [(preprocess_fn(img), fname) for img, fname in source.iter_simple_forgery(user)]
        if simple_data:
            simple_imgs, simple_fnames = zip(*simple_data)
            cnt = len(simple_imgs)
            x[N: N + cnt] = simple_imgs
            yforg[N: N + cnt] = 2
            y[N: N + cnt] = i
            used_files += list(simple_fnames)
            N += cnt

        user_mapping[i] = user

    if N != max_sigs:
        x.resize((N, 1, H, W), refcheck=False)
        y.resize(N, refcheck=False)
        yforg.resize(N, refcheck=False)
    else:
        x = np.expand_dims(x, 1)

    return x, y, yforg, user_mapping, np.array(used_files)


def get_subset(data: Tuple[np.ndarray, ...],
               subset: Union[list, range],
               y_idx: int = 1) -> Tuple[np.ndarray, ...]:
    """Return rows of data whose user label (data[y_idx]) is in subset."""
    mask = np.isin(data[y_idx], subset)
    return tuple(d[mask] for d in data)


def remove_forgeries(data: Tuple[np.ndarray, ...],
                     forg_idx: int = 2) -> Tuple[np.ndarray, ...]:
    """Drop forged samples from a dataset tuple."""
    mask = data[forg_idx] == 0
    return tuple(d[mask] for d in data)
