import re
from copy import deepcopy
from typing import Any

import kornia
import torch
from torch.utils.data.dataloader import default_collate


class CollateBase:
    def __init__(self, item_keys: str | list[str] = ".*", delete_original=False):
        self.item_keys = item_keys if isinstance(item_keys, list) else [item_keys]
        self.item_keys = [re.compile(key) for key in self.item_keys]

        self.delete_original = delete_original

        self.item_keys_cached = []

    def __call__(
        self, items_list: list[dict[str, Any]] | dict[str, Any]
    ) -> dict[str, Any]:
        assert len(items_list) > 0, "items_list must have at least one item"

        if isinstance(items_list, list):
            items = {
                key: [item[key] for item in items_list] for key in items_list[0].keys()
            }
        else:
            items = items_list

        keys = self.match_keys(list(items.keys()))

        items_new = self.do_collate({key: items[key] for key in keys})
        if self.delete_original:
            for key in keys:
                del items[key]

        for key in items_new.keys():
            items[key] = items_new[key]

        return items

    def match_keys(self, keys: list[str]) -> list[str]:
        if not self.item_keys_cached:
            for key in keys:
                for item_key in self.item_keys:
                    if re.match(item_key, key):
                        self.item_keys_cached.append(key)
                        break

        return self.item_keys_cached

    def do_collate(self, item: dict[str, Any]):
        raise NotImplementedError


class BatchCollate(CollateBase):
    def make_into_tensor(self, items):
        if isinstance(items, torch.Tensor):
            return items.float()
        if isinstance(items, str):
            return items
        if isinstance(items, list):
            converted_items = [self.make_into_tensor(item) for item in items]
            if all(isinstance(item, torch.Tensor) for item in converted_items):
                return torch.stack(converted_items)
            return converted_items
        if isinstance(items, dict):
            return {key: self.make_into_tensor(item) for key, item in items.items()}

        return torch.tensor(items).float()

    def do_collate(self, items):
        result = self.make_into_tensor(items)
        return result


class DeleteKeys(CollateBase):
    def __init__(self, keys: list[str]):
        super().__init__(keys, True)

    def do_collate(self, items):
        return {}


class ListCollate(CollateBase):
    def __init__(
        self, collates: list[CollateBase], item_keys=".*", delete_original=False
    ):
        super().__init__(item_keys, delete_original)
        self.collates = collates

    def do_collate(self, items):
        for collate in self.collates:
            items = collate(items)

        return items


class RandomSaltAndPepperNoise(CollateBase):
    def __init__(
        self, amount=(0.01, 0.06), salt_vs_pepper=(0.4, 0.6), p=0.5, item_keys=".*"
    ):
        super().__init__(item_keys)

        self.noise_transform = kornia.augmentation.RandomSaltAndPepperNoise(
            amount=amount,
            salt_vs_pepper=salt_vs_pepper,
            p=p,
            keepdim=True,
        )

    def do_collate(self, images):
        for key, image in images.items():
            if not isinstance(image, torch.Tensor):
                raise TypeError("Input image must be a torch.Tensor")

            images[key] = self.noise_transform(image)
        return images


class RandomRotate(CollateBase):
    def __init__(self, max_angle=20, share_rotations=False, item_keys=".*"):
        super().__init__(item_keys)

        self.max_angle = torch.tensor(max_angle).float()
        self.share_rotations = share_rotations

    def do_collate(self, images):
        if self.share_rotations:
            angle = torch.normal(mean=0, std=self.max_angle)
            return {
                key: kornia.geometry.transform.rotate(image, angle)
                for key, image in images.items()
            }

        for key, image in images.items():
            angle = torch.normal(mean=0, std=self.max_angle)
            images[key] = kornia.geometry.transform.rotate(image, angle)
        return images


class RandomVerticalFlip(CollateBase):
    def __init__(self, p=0.5, share_flip=False, item_keys=".*"):
        super().__init__(item_keys)
        self.p = p
        self.share_flip = share_flip

    def do_collate(self, images):
        if self.share_flip:
            do_flip = torch.rand(1).item() < self.p
            if not do_flip:
                return images
            return {
                key: kornia.geometry.transform.vflip(image)
                for key, image in images.items()
            }

        for key, image in images.items():
            do_flip = torch.rand(1).item() < self.p
            if do_flip:
                images[key] = kornia.geometry.transform.vflip(image)
        return images


class ColorJitter(CollateBase):
    def __init__(self, br=0.5, sat=0.5, p=0.5, item_keys=".*"):
        super().__init__(item_keys)

        self.color_jitter = kornia.augmentation.ColorJitter(
            brightness=br, saturation=sat, p=p, keepdim=True
        )

    def do_collate(self, images):
        return {key: self.color_jitter(image) for key, image in images.items()}


class GaussianBlur(CollateBase):
    def __init__(self, kernel_size=(3, 3), sigma=(1, 10), p=0.5, item_keys=".*"):
        super().__init__(item_keys)

        self.random_gblur = kornia.augmentation.RandomGaussianBlur(
            kernel_size=kernel_size, sigma=sigma, p=p, keepdim=True
        )

    def do_collate(self, images):
        return {key: self.random_gblur(image) for key, image in images.items()}


class Clipping(CollateBase):
    def __init__(self, min_val=0, max_val=1, item_keys=".*"):
        super().__init__(item_keys)
        self.min_val = min_val
        self.max_val = max_val

    def do_collate(self, images):
        return {
            key: torch.clamp(image, self.min_val, self.max_val)
            for key, image in images.items()
        }


class Normalization(CollateBase):
    def __init__(self, mean=0.5, std=0.5, item_keys=".*"):
        super().__init__(item_keys)
        self.mean = mean
        self.std = std

    def do_collate(self, images):
        return {key: (image - self.mean) / self.std for key, image in images.items()}


class MaxCollate(CollateBase):
    def do_collate(self, items):
        return {key: torch.max(item) for key, item in items.items()}


class ConcatenateCollate(CollateBase):
    def __init__(self, new_key: str, dim=1, item_keys=".*", delete_original=True):
        super().__init__(item_keys, delete_original)

        self.new_key = new_key
        self.dim = dim

    def do_collate(self, items):
        item_list = [items[key] for key in sorted(items.keys())]
        return {self.new_key: torch.cat(item_list, dim=self.dim)}


class OutlierCollate(CollateBase):
    def __init__(self, thresholds, item_keys=".*"):
        super().__init__(item_keys)

        self.thresholds = thresholds

    def do_collate(self, items):
        for key in items.keys():
            items[key][items[key] < self.thresholds[0]] = self.thresholds[0]
            items[key][items[key] > self.thresholds[1]] = self.thresholds[1]
        return items


class ScaleData(CollateBase):
    def __init__(self, min_val=0, max_val=1, center=0.5, item_keys=".*"):
        super().__init__(item_keys)
        self.min_val = min_val
        self.max_val = max_val
        self.center = center

    def do_collate(self, images):
        return {key: self.scale_data(image) for key, image in images.items()}

    def scale_data(self, data):
        if self.center == 0.5:
            return (data - self.min_val) / (self.max_val - self.min_val) * 2 - 1
        elif self.center == 0:
            return (data - self.min_val) / (self.max_val - self.min_val)
        else:
            raise ValueError("center must be 0 or 0.5")
