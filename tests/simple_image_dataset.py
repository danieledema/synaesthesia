from src.synaesthesia.base_sensors.image_dataset import ImageDataset


class SimpleImageDataset(ImageDataset):
    @property
    def id(self):
        return "camera"

    def get_machine_name(self) -> str:
        return "top_camera"

    def parse_filename(self, filename) -> int:
        return int(filename.stem)
