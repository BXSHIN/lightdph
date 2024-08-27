from typing import Iterable, Callable


class Compose():  # pylint: disable=too-few-public-methods
    def __init__(self, transforms: Iterable[Callable]):
        self.transforms = transforms

    def __call__(self, dat):
        for t in self.transforms:
            dat = t(dat)
        return dat

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string
