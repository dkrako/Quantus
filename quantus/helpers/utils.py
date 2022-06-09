"""This module contains the utils functions of the library."""
import re
import random
from importlib import util
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from skimage.segmentation import slic, felzenszwalb

if util.find_spec("torch"):
    import torch
    from ..helpers.pytorch_model import PyTorchModel
if util.find_spec("tensorflow"):
    import tensorflow as tf
    from ..helpers.tf_model import TensorFlowModel

from ..typing import Patch
from ..helpers.model_interface import ModelInterface


def get_superpixel_segments(img: np.ndarray, segmentation_method: str) -> np.ndarray:
    """Given an image, return segments or so-called 'super-pixels' segments i.e., an 2D mask with segment labels."""
    if img.ndim != 3:
        raise ValueError(
            "Make sure that x is 3 dimensional e.g., (3, 224, 224) to calculate super-pixels."
            f" shape: {img.shape}"
        )
    if segmentation_method not in ["slic", "felzenszwalb"]:
        raise ValueError(
            "'segmentation_method' must be either 'slic' or 'felzenszwalb'."
        )

    if segmentation_method == "slic":
        return slic(img, start_label=0)
    elif segmentation_method == "felzenszwalb":
        return felzenszwalb(
            img,
        )


def get_baseline_value(
    choice: Union[float, int, str, None],
    arr: np.ndarray,
    patch: Optional[np.ndarray] = None,
    **kwargs,
) -> float:
    """Get the baseline value (float) to fill the array with."""
    if choice is None:
        assert (
            ("perturb_baseline" in kwargs)
            or ("fixed_values" in kwargs)
            or ("constant_value" in kwargs)
            or ("input_shift" in kwargs)
        ), (
            "Specify"
            "a 'perturb_baseline', 'fixed_values', 'constant_value' or 'input_shift' e.g., 0.0 or 'black' for "
            "pixel replacement or 'baseline_values' containing an array with one value per index for replacement."
        )

    if "fixed_values" in kwargs:
        return kwargs["fixed_values"]
    if isinstance(choice, (float, int)):
        return choice
    elif isinstance(choice, str):
        valid_choices = ["mean", "black", "min", "white", "max",
                         "random", "uniform", "saltnpepper", "gaussian",
                         "neighbourhood_mean", "neighbourhood_random_min_max"]
        if choice not in valid_choices:
            raise ValueError(f"Ensure that 'choice'(str) is in {valid_choices}")
        elif choice == "mean":
            return float(arr.mean())
        elif choice == "random":
            return float(random.random())
        elif choice == "uniform":
            return float(np.random.uniform(arr.min(), arr.max()))
        elif choice == "black" or choice == "min":
            return float(arr.min())
        elif choice == "white" or choice == "max":
            return float(arr.max())
        elif choice == "saltnpepper":
            return float(random.choice([arr.min(), arr.max()]))
        elif choice == "gaussian":
            return float(random.gauss(mu=arr.mean(), sigma=arr.std()))
        elif choice == "neighbourhood_mean":
            if patch is None:
                raise ValueError("patch must not be None for neighbourhood_mean")
            return float(patch.mean())
        elif choice == "neighbourhood_random_min_max":
            if patch is None:
                raise ValueError("patch must not be None for neighbourhood_random_min_max")
            return float(random.uniform(patch.min(), patch.max())),
    else:
        raise ValueError(
            "Specify 'perturb_baseline' or 'constant_value' as a string, integer or float."
        )


def get_name(str: str):
    """Get the name of the class object."""
    if str.isupper():
        return str
    return " ".join(re.sub(r"([A-Z])", r" \1", str).split())


def get_features_in_step(max_steps_per_input: int, input_shape: Tuple[int, ...]):
    """Get the number of features in the iteration."""
    return np.prod(input_shape) / max_steps_per_input


def filter_compatible_patch_sizes(perturb_patch_sizes: list, img_size: int) -> list:
    """Remove patch sizes that are not compatible with input size."""
    return [i for i in perturb_patch_sizes if img_size % i == 0]


def infer_channel_first(x: np.array):
    """
    Infer if the channels are first.

    For 1d input:

        Assumes
            nr_channels < sequence_length
        Returns
            True if input shape is (nr_batch, nr_channels, sequence_length).
            False if input shape is (nr_batch, sequence_length, nr_channels).
            An error is raised if the two last dimensions are equal.

    For 2d input:

        Assumes
            nr_channels < img_width and nr_channels < img_height
        Returns
            True if input shape is (nr_batch, nr_channels, img_width, img_height).
            False if input shape is (nr_batch, img_width, img_height, nr_channels).
            An error is raised if the three last dimensions are equal.

    For higher dimensional input an error is raised.
    """
    err_msg = "Ambiguous input shape. Cannot infer channel-first/channel-last order."

    if len(np.shape(x)) == 3:
        if np.shape(x)[-2] < np.shape(x)[-1]:
            return True
        elif np.shape(x)[-2] > np.shape(x)[-1]:
            return False
        else:
            raise ValueError(err_msg)

    elif len(np.shape(x)) == 4:
        if np.shape(x)[-1] < np.shape(x)[-2] and np.shape(x)[-1] < np.shape(x)[-3]:
            return False
        if np.shape(x)[-3] < np.shape(x)[-1] and np.shape(x)[-3] < np.shape(x)[-2]:
            return True
        raise ValueError(err_msg)

    else:
        raise ValueError(
            "Only batched 1d and 2d multi-channel input dimensions supported."
        )


def make_channel_first(x: np.array, channel_first: bool = False):
    """Reshape batch to channel first."""
    if channel_first:
        return x

    if len(np.shape(x)) == 4:
        return np.moveaxis(x, -1, -3)
    elif len(np.shape(x)) == 3:
        return np.moveaxis(x, -1, -2)
    else:
        raise ValueError(
            "Only batched 1d and 2d multi-channel input dimensions supported."
        )


def make_channel_last(x: np.array, channel_first: bool = True):
    """Reshape batch to channel last."""
    if not channel_first:
        return x

    if len(np.shape(x)) == 4:
        return np.moveaxis(x, -3, -1)
    elif len(np.shape(x)) == 3:
        return np.moveaxis(x, -2, -1)
    else:
        raise ValueError(
            "Only batched 1d and 2d multi-channel input dimensions supported."
        )


def get_wrapped_model(
        model: ModelInterface,
        channel_first: bool,
        predict_kwargs: Optional[Dict[str, Any]] = None,
) -> ModelInterface:
    """
    Identifies the type of a model object and wraps the model in an appropriate interface.

    Returns
        A wrapped ModelInterface model.
    """
    if isinstance(model, tf.keras.Model):
        return TensorFlowModel(model, channel_first, predict_kwargs)
    if isinstance(model, torch.nn.modules.module.Module):
        return PyTorchModel(model, channel_first, predict_kwargs)
    raise ValueError(
        "Model needs to be tf.keras.Model or torch.nn.modules.module.Module."
    )


def conv2D_numpy(
    x: np.array,
    kernel: np.array,
    stride: int,
    padding: int,
    groups: int,
    pad_output: bool = False,
) -> np.array:
    """
    Computes 2D convolution in NumPy.

    Assumes
        Shape of x is [C_in, H, W] with C_in = input channels and H, W input height and weight, respectively
        Shape of kernel is [C_out, C_in/groups, K, K] with C_out = output channels and K = kernel size
    """

    # Pad input
    x = np.pad(x, [(0, 0), (padding, padding), (padding, padding)], mode="constant")

    # Get shapes
    c_in, height, width = x.shape
    c_out, kernel_size = kernel.shape[0], kernel.shape[2]

    # Handle groups
    assert c_in % groups == 0
    assert c_out % groups == 0
    assert kernel.shape[1] * groups == c_in
    c_in_g = c_in // groups
    c_out_g = c_out // groups

    # Build output
    output_height = (height - kernel_size) // stride + 1
    output_width = (width - kernel_size) // stride + 1
    output = np.zeros((c_out, output_height, output_width)).astype(x.dtype)

    # TODO: improve efficiency, less loops
    for g in range(groups):
        for c in range(c_out_g * g, c_out_g * (g + 1)):
            for h in range(output_height):
                for w in range(output_width):
                    output[c][h][w] = np.multiply(
                        x[
                            c_in_g * g : c_in_g * (g + 1),
                            h * stride : h * stride + kernel_size,
                            w * stride : w * stride + kernel_size,
                        ],
                        kernel[c, :, :, :],
                    ).sum()

    if pad_output:
        if stride != 1 or padding != 0:
            raise NotImplementedError()
        padwidth = (kernel_size - 1) // 2
        output = np.pad(
            output,
            (
                (0, 0),
                (padwidth + padwidth % 2, padwidth),
                (padwidth + padwidth % 2, padwidth),
            ),
            mode="edge",
        )

    return output


def create_patch(
    patch_size: Union[int, Sequence[int]], coords: Sequence[int], expand_first_dim: bool
) -> Patch:
    """
    Create a patch from patch size and coordinates.
    expand_first_dim: set to True if you want to add one ':'-slice at the beginning.
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size,)
    if isinstance(coords, int):
        coords = (coords,)

    patch_size = np.array(patch_size)
    coords = tuple(coords)

    if len(patch_size) == 1 and len(coords) != 1:
        patch_size = tuple(patch_size for _ in coords)
    elif patch_size.ndim != 1:
        raise ValueError("patch_size has to be either a scalar or a 1d-sequence")
    elif len(patch_size) != len(coords):
        raise ValueError(
            "patch_size sequence length does not match coords length"
            f" (len(patch_size) != len(coords))"
        )
    # make sure that each element in tuple is integer
    patch_size = tuple(int(patch_size_dim) for patch_size_dim in patch_size)

    patch = [
        slice(coord, coord + patch_size_dim)
        for coord, patch_size_dim in zip(coords, patch_size)
    ]
    # Prepend slice for all channels.
    if expand_first_dim:
        patch = [slice(None), *patch]

    return tuple(patch)


# TODO: this isn't actually used
def transform_patch_to_indices(patch, arr_shape): # TODO: add typehint
    mask = np.zeros(arr_shape, dtype=bool)
    mask[patch] = True
    return np.where(mask)


def transform_patches_to_indices(patches, arr_shape): #TODO: add typehint
    # Check if list is batched.
    if isinstance(patches[0], list):
        # Recursive call for each patch list in batch.
        return np.array([
            transform_patches_to_indices(patches_instance, arr_shape[1:])
            for patches_instance in patches
        ])

    return np.array([
        transform_patch_to_indices(patch, arr_shape)
        for patch in patches
    ])


def expand_attribution_channel(a: np.ndarray, x: np.ndarray):
    """Expand additional channel dimension for attributions if needed."""
    if a.shape[0] != x.shape[0]:
        raise ValueError(
            f"a and x must have same number of batches ({a.shape[0]} != {x.shape[0]})"
        )
    if a.ndim > x.ndim:
        raise ValueError(f"a must not have greater ndim than x ({a.ndim} > {x.ndim})")
    if a.ndim < x.ndim - 1:
        raise ValueError(
            f"a can have at max one dimension less than x ({a.ndim} < {x.ndim} - 1)"
        )

    if a.ndim == x.ndim:
        return a
    elif a.ndim == x.ndim - 1:
        return np.expand_dims(a, axis=1)


def get_nr_patches(
    patch_size: Union[int, Sequence[int]], shape: Tuple[int, ...], overlap: bool = False
) -> int:
    """Get number of patches for given shape."""
    if isinstance(patch_size, int):
        patch_size = (patch_size,)
    patch_size = np.array(patch_size)

    if len(patch_size) == 1 and len(shape) != 1:
        patch_size = tuple(patch_size for _ in shape)
    elif patch_size.ndim != 1:
        raise ValueError("patch_size has to be either a scalar or a 1d-sequence")
    elif len(patch_size) != len(shape):
        raise ValueError(
            "patch_size sequence length does not match shape length"
            f" (len(patch_size) != len(shape))"
        )
    patch_size = tuple(patch_size)

    return np.prod(shape) // np.prod(patch_size)


def pad_array(arr: np.ndarray, pad_width: int, mode: str,
              omit_first_axis: bool = True, batched: bool = True,
) -> np.ndarray:
    """To allow for any patch_size we add padding to the array."""
    pad_width_list = [(pad_width, pad_width)] * arr.ndim
    
    if batched:
        pad_width_list[0] = (0, 0)
    if omit_first_axis:
        if batched:
            pad_width_list[1] = (0, 0)
        else:
            pad_width_list[0] = (0, 0)
            
    arr_pad = np.pad(arr, pad_width_list, mode="constant")
    return arr_pad


def unpad_array(arr: np.ndarray, pad_width: int,
                omit_first_axis: bool = True, batched: bool = True,
) -> np.ndarray:
    """Remove padding from the array."""
    unpad_slice = [
        slice(pad_width, arr.shape[axis] - pad_width)
        for axis, _ in enumerate(arr.shape)
    ]

    if batched:
        unpad_slice[0] = slice(None)
    if omit_first_axis:
        if batched:
            unpad_slice[1] = slice(None)
        else:
            unpad_slice[0] = slice(None)
        
    return arr[tuple(unpad_slice)]


