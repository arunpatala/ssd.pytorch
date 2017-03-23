"""SSD Dataset factory

Returns the corresponding dataset object for use in the SSD network

Ellis Brown
"""

from . import SSDVOC, RGB_MEANS, AnnotationTransform
from .. import VOCroot, base_transform, TrainTransform


def ssd_dataset(dataset, image_set, ssd_dim):
    """SSD Dataset factory

    Arguments:
        dataset (string): dataset to use (eg. 'VOC')
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        ssd_dim (int): dimension of ssd (300, or 512)
    Return:
        the specified dataset object for use in the SSD network
    """
    if dataset.lower() == 'voc':
        trans = base_transform(ssd_dim, RGB_MEANS) if \
            image_set.lower() == "train" else TrainTransform()

        return SSDVOC(VOCroot, image_set, trans, AnnotationTransform())

    # if parameters do not match, throw exception
    assert 0, "Bad dataset creation.\ndataset: " + dataset +  \
        "\nimage_set: " + image_set + "\nssd_dim: " + str(ssd_dim)
