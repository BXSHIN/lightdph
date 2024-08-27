import logging

from torchvision.datasets import ImageFolder

# Initiate Logger
logger = logging.getLogger(__name__)


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        try:
            img = self.loader(path)
        except Exception as e:
            logger.error(e, path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path
