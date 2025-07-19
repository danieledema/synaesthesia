import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import kornia
import torch


class CollateBase(ABC):
    """
    Abstract base class for custom collate functions.

    This class provides a framework for creating custom collate functions that can
    selectively apply transformations to specific keys in the input data based on
    regular expression patterns.

    Args:
        item_keys: String or list of strings containing regex patterns to match keys.
                  Defaults to ".*" (matches all keys).
        delete_original: Whether to delete the original keys after transformation.
                        Defaults to False.
    """

    def __init__(
        self, item_keys: Union[str, List[str]] = ".*", delete_original: bool = False
    ) -> None:
        self.item_keys = item_keys if isinstance(item_keys, list) else [item_keys]
        self.item_keys_compiled = [re.compile(key) for key in self.item_keys]
        self.delete_original = delete_original
        self._item_keys_cache: Dict[str, List[str]] = {}

    def __call__(
        self, items_list: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply the collate function to the input data.

        Args:
            items_list: Either a list of dictionaries (batch) or a single dictionary.

        Returns:
            Dictionary with transformed data.

        Raises:
            AssertionError: If items_list is empty.
        """
        if not items_list:
            raise ValueError("items_list must have at least one item")

        # Convert list of dicts to dict of lists for batch processing
        if isinstance(items_list, list):
            if not items_list:
                raise ValueError("Empty items list provided")
            items = {
                key: [item[key] for item in items_list] for key in items_list[0].keys()
            }
        else:
            items = items_list

        # Get matching keys and apply transformation
        keys = self._match_keys(list(items.keys()))
        if not keys:
            return items

        matched_items = {key: items[key] for key in keys}
        transformed_items = self.do_collate(matched_items)

        # Update the items dictionary
        if self.delete_original:
            for key in keys:
                items.pop(key, None)

        items.update(transformed_items)
        return items

    def _match_keys(self, keys: List[str]) -> List[str]:
        """
        Match keys against the compiled regex patterns with caching.

        Args:
            keys: List of keys to match against patterns.

        Returns:
            List of matched keys.
        """
        keys_tuple = tuple(sorted(keys))

        if keys_tuple not in self._item_keys_cache:
            matched_keys = []
            for key in keys:
                for pattern in self.item_keys_compiled:
                    if pattern.match(key):
                        matched_keys.append(key)
                        break
            self._item_keys_cache[keys_tuple] = matched_keys

        return self._item_keys_cache[keys_tuple]

    @abstractmethod
    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to be implemented by subclasses.

        Args:
            items: Dictionary of items to transform.

        Returns:
            Dictionary of transformed items.
        """
        pass


class BatchCollate(CollateBase):
    """
    Collate function that converts items to tensors and handles batching.

    This collate function handles conversion of various data types to PyTorch tensors,
    including proper handling of NaN values and tensor stacking for batches.
    """

    def _make_into_tensor(self, items: Any) -> Any:
        """
        Convert items to tensors recursively.

        Args:
            items: Items to convert to tensors.

        Returns:
            Converted items as tensors or appropriate data structures.
        """
        if isinstance(items, torch.Tensor):
            return items.float()

        if isinstance(items, str):
            return items

        if items is None:
            return torch.tensor(float("nan"))

        if isinstance(items, list):
            converted_items = [self._make_into_tensor(item) for item in items]

            # If not all items are tensors, return as list
            if not all(isinstance(item, torch.Tensor) for item in converted_items):
                return converted_items

            # Handle NaN values by ensuring consistent shapes
            if any(torch.isnan(item).any() for item in converted_items):
                valid_shapes = [
                    item.shape
                    for item in converted_items
                    if not torch.isnan(item).any()
                ]

                if not valid_shapes:
                    return converted_items

                target_shape = valid_shapes[0]
                if not all(shape == target_shape for shape in valid_shapes):
                    raise ValueError("All tensors must have the same shape")

                # Replace NaN tensors with properly shaped NaN tensors
                converted_items = [
                    item
                    if not torch.isnan(item).any()
                    else torch.full(target_shape, float("nan"))
                    for item in converted_items
                ]

            return torch.stack(converted_items)

        if isinstance(items, dict):
            return {key: self._make_into_tensor(item) for key, item in items.items()}

        return torch.tensor(items).float()

    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert items to tensors.

        Args:
            items: Dictionary of items to convert.

        Returns:
            Dictionary with items converted to tensors.
        """
        return {key: self._make_into_tensor(value) for key, value in items.items()}


class DeleteKeys(CollateBase):
    """
    Collate function that deletes specified keys from the data.

    Args:
        keys: List of key patterns to delete.
    """

    def __init__(self, keys: List[str]) -> None:
        super().__init__(keys, delete_original=True)

    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """Delete the matched keys by returning empty dict."""
        return {}


class ListCollate(CollateBase):
    """
    Collate function that applies a sequence of collate functions.

    Args:
        collates: List of collate functions to apply in sequence.
        item_keys: Key patterns to match. Defaults to ".*".
        delete_original: Whether to delete original keys. Defaults to False.
    """

    def __init__(
        self,
        collates: List[CollateBase],
        item_keys: Union[str, List[str]] = ".*",
        delete_original: bool = False,
    ) -> None:
        super().__init__(item_keys, delete_original)
        self.collates = collates

    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all collate functions in sequence.

        Args:
            items: Dictionary of items to transform.

        Returns:
            Dictionary with all transformations applied.
        """
        result = items
        for collate in self.collates:
            result = collate(result)
        return result


class ImageAugmentationMixin:
    """Mixin class for common image augmentation functionality."""

    def _validate_image_tensor(self, image: Any, key: str) -> torch.Tensor:
        """
        Validate that the input is a proper image tensor.

        Args:
            image: Input to validate.
            key: Key name for error reporting.

        Returns:
            Validated tensor.

        Raises:
            TypeError: If input is not a tensor.
            ValueError: If tensor doesn't have proper dimensions.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"Input for key '{key}' must be a torch.Tensor, got {type(image)}"
            )

        if image.dim() < 2:
            raise ValueError(
                f"Image tensor for key '{key}' must have at least 2 dimensions"
            )

        return image


class RandomSaltAndPepperNoise(CollateBase, ImageAugmentationMixin):
    """
    Add random salt and pepper noise to images using Kornia.

    Args:
        amount: Tuple of (min, max) noise amount. Defaults to (0.01, 0.06).
        salt_vs_pepper: Tuple of (min, max) salt vs pepper ratio. Defaults to (0.4, 0.6).
        p: Probability of applying the transformation. Defaults to 0.5.
        item_keys: Key patterns to match. Defaults to ".*".
    """

    def __init__(
        self,
        amount: Tuple[float, float] = (0.01, 0.06),
        salt_vs_pepper: Tuple[float, float] = (0.4, 0.6),
        p: float = 0.5,
        item_keys: Union[str, List[str]] = ".*",
    ) -> None:
        super().__init__(item_keys)
        self.noise_transform = kornia.augmentation.RandomSaltAndPepperNoise(
            amount=amount,
            salt_vs_pepper=salt_vs_pepper,
            p=p,
            keepdim=True,
        )

    def do_collate(self, images: Dict[str, Any]) -> Dict[str, Any]:
        """Apply salt and pepper noise to images."""
        result = {}
        for key, image in images.items():
            validated_image = self._validate_image_tensor(image, key)
            result[key] = self.noise_transform(validated_image)
        return result


class RandomRotate(CollateBase, ImageAugmentationMixin):
    """
    Randomly rotate images using Kornia.

    Args:
        max_angle: Maximum rotation angle in degrees. Defaults to 20.
        share_rotations: Whether to use the same rotation for all images. Defaults to False.
        item_keys: Key patterns to match. Defaults to ".*".
    """

    def __init__(
        self,
        max_angle: float = 20,
        share_rotations: bool = False,
        item_keys: Union[str, List[str]] = ".*",
    ) -> None:
        super().__init__(item_keys)
        self.max_angle = max_angle
        self.share_rotations = share_rotations

    def do_collate(self, images: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random rotation to images."""
        if self.share_rotations:
            angle = torch.normal(
                mean=torch.tensor(0.0), std=torch.tensor(self.max_angle)
            )
            return {
                key: kornia.geometry.transform.rotate(
                    self._validate_image_tensor(image, key), angle
                )
                for key, image in images.items()
            }

        result = {}
        for key, image in images.items():
            validated_image = self._validate_image_tensor(image, key)
            angle = torch.normal(
                mean=torch.tensor(0.0), std=torch.tensor(self.max_angle)
            )
            result[key] = kornia.geometry.transform.rotate(validated_image, angle)
        return result


class RandomVerticalFlip(CollateBase, ImageAugmentationMixin):
    """
    Randomly flip images vertically using Kornia.

    Args:
        p: Probability of flipping. Defaults to 0.5.
        share_flip: Whether to use the same flip decision for all images. Defaults to False.
        item_keys: Key patterns to match. Defaults to ".*".
    """

    def __init__(
        self,
        p: float = 0.5,
        share_flip: bool = False,
        item_keys: Union[str, List[str]] = ".*",
    ) -> None:
        super().__init__(item_keys)
        self.p = p
        self.share_flip = share_flip

    def do_collate(self, images: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random vertical flip to images."""
        if self.share_flip:
            do_flip = torch.rand(1).item() < self.p
            if not do_flip:
                return images
            return {
                key: kornia.geometry.transform.vflip(
                    self._validate_image_tensor(image, key)
                )
                for key, image in images.items()
            }

        result = {}
        for key, image in images.items():
            validated_image = self._validate_image_tensor(image, key)
            if torch.rand(1).item() < self.p:
                result[key] = kornia.geometry.transform.vflip(validated_image)
            else:
                result[key] = validated_image
        return result


class ColorJitter(CollateBase, ImageAugmentationMixin):
    """
    Apply random color jittering using Kornia.

    Args:
        brightness: Brightness jitter factor. Defaults to 0.5.
        saturation: Saturation jitter factor. Defaults to 0.5.
        p: Probability of applying the transformation. Defaults to 0.5.
        item_keys: Key patterns to match. Defaults to ".*".
    """

    def __init__(
        self,
        brightness: float = 0.5,
        saturation: float = 0.5,
        p: float = 0.5,
        item_keys: Union[str, List[str]] = ".*",
    ) -> None:
        super().__init__(item_keys)
        self.color_jitter = kornia.augmentation.ColorJitter(
            brightness=brightness, saturation=saturation, p=p, keepdim=True
        )

    def do_collate(self, images: Dict[str, Any]) -> Dict[str, Any]:
        """Apply color jittering to images."""
        return {
            key: self.color_jitter(self._validate_image_tensor(image, key))
            for key, image in images.items()
        }


class GaussianBlur(CollateBase, ImageAugmentationMixin):
    """
    Apply random Gaussian blur using Kornia.

    Args:
        kernel_size: Blur kernel size. Defaults to (3, 3).
        sigma: Blur sigma range. Defaults to (1, 10).
        p: Probability of applying the transformation. Defaults to 0.5.
        item_keys: Key patterns to match. Defaults to ".*".
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        sigma: Tuple[float, float] = (1, 10),
        p: float = 0.5,
        item_keys: Union[str, List[str]] = ".*",
    ) -> None:
        super().__init__(item_keys)
        self.random_gblur = kornia.augmentation.RandomGaussianBlur(
            kernel_size=kernel_size, sigma=sigma, p=p, keepdim=True
        )

    def do_collate(self, images: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Gaussian blur to images."""
        return {
            key: self.random_gblur(self._validate_image_tensor(image, key))
            for key, image in images.items()
        }


class Clipping(CollateBase):
    """
    Clip tensor values to a specified range.

    Args:
        min_val: Minimum value for clipping. Defaults to 0.
        max_val: Maximum value for clipping. Defaults to 1.
        item_keys: Key patterns to match. Defaults to ".*".
    """

    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 1,
        item_keys: Union[str, List[str]] = ".*",
    ) -> None:
        super().__init__(item_keys)
        self.min_val = min_val
        self.max_val = max_val

    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """Clip values to the specified range."""
        return {
            key: torch.clamp(item, self.min_val, self.max_val)
            for key, item in items.items()
        }


class Normalization(CollateBase):
    """
    Normalize tensors using mean and standard deviation.

    Args:
        mean: Mean for normalization. Defaults to 0.5.
        std: Standard deviation for normalization. Defaults to 0.5.
        item_keys: Key patterns to match. Defaults to ".*".
    """

    def __init__(
        self,
        mean: float = 0.5,
        std: float = 0.5,
        item_keys: Union[str, List[str]] = ".*",
    ) -> None:
        super().__init__(item_keys)
        self.mean = mean
        self.std = std

    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize items using the specified mean and std."""
        return {key: (item - self.mean) / self.std for key, item in items.items()}


class MaxCollate(CollateBase):
    """
    Compute the maximum value for each tensor.
    """

    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """Compute maximum values."""
        return {
            key: torch.max(item) if isinstance(item, torch.Tensor) else item
            for key, item in items.items()
        }


class ConcatenateCollate(CollateBase):
    """
    Concatenate tensors along a specified dimension.

    Args:
        new_key: Key name for the concatenated result.
        dim: Dimension along which to concatenate. Defaults to 1.
        item_keys: Key patterns to match. Defaults to ".*".
        delete_original: Whether to delete original keys. Defaults to True.
    """

    def __init__(
        self,
        new_key: str,
        dim: int = 1,
        item_keys: Union[str, List[str]] = ".*",
        delete_original: bool = True,
    ) -> None:
        super().__init__(item_keys, delete_original)
        self.new_key = new_key
        self.dim = dim

    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """Concatenate tensors along the specified dimension."""
        if not items:
            return {}

        # Sort keys for consistent ordering
        sorted_keys = sorted(items.keys())
        tensors = [items[key] for key in sorted_keys]

        # Validate that all items are tensors
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("All items must be tensors for concatenation")

        return {self.new_key: torch.cat(tensors, dim=self.dim)}


class OutlierCollate(CollateBase):
    """
    Clip values outside specified thresholds (outlier removal).

    Args:
        thresholds: Tuple of (min_threshold, max_threshold).
        item_keys: Key patterns to match. Defaults to ".*".
    """

    def __init__(
        self, thresholds: Tuple[float, float], item_keys: Union[str, List[str]] = ".*"
    ) -> None:
        super().__init__(item_keys)
        self.min_threshold, self.max_threshold = thresholds

    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """Clip outlier values."""
        result = {}
        for key, item in items.items():
            if isinstance(item, torch.Tensor):
                clipped = torch.clamp(item, self.min_threshold, self.max_threshold)
                result[key] = clipped
            else:
                result[key] = item
        return result


class ScaleData(CollateBase):
    """
    Scale data to a specified range based on min/max values and center point.

    Args:
        min_val: Minimum value of the input range. Defaults to 0.
        max_val: Maximum value of the input range. Defaults to 1.
        center: Center point for scaling (0 or 0.5). Defaults to 0.5.
        item_keys: Key patterns to match. Defaults to ".*".
    """

    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 1,
        center: float = 0.5,
        item_keys: Union[str, List[str]] = ".*",
    ) -> None:
        super().__init__(item_keys)
        self.min_val = min_val
        self.max_val = max_val
        self.center = center

        if center not in (0, 0.5):
            raise ValueError("center must be 0 or 0.5")

    def do_collate(self, items: Dict[str, Any]) -> Dict[str, Any]:
        """Scale data according to the specified parameters."""
        return {key: self._scale_data(item) for key, item in items.items()}

    def _scale_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Scale data to the specified range.

        Args:
            data: Input tensor to scale.

        Returns:
            Scaled tensor.
        """
        normalized = (data - self.min_val) / (self.max_val - self.min_val)

        if self.center == 0.5:
            return normalized * 2 - 1  # Scale to [-1, 1]
        else:  # center == 0
            return normalized  # Scale to [0, 1]
