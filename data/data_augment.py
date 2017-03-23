"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

TODO: explore https://github.com/ncullen93/torchsample/blob/master/torchsample/transforms
    for any useful tranformations
TODO: implement data_augment for training

Ellis Brown
"""

import random
import torch
from torchvision import transforms

from box_utils import jaccard
# import torch_transforms


def crop(img, boxes, labels, mode):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    while True:
        width, height = img.size

        if mode is None:
            return img, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            w = random.randrange(int(0.3 * width), width)
            h = random.randrange(int(0.3 * height), height)

            # aspect ratio b/t .5 & 2
            if h / w < 0.5 or h / w > 2:
                continue

            rect = torch.Tensor([[random.randrange(width - w),
                                  random.randrange(height - h),
                                  w, h]])

            overlap = jaccard(boxes, rect)
            if overlap.min() < min_iou and max_iou < overlap.max():
                continue

            img = img[rect[0, 1]:rect[0, 3], rect[0, 0]:rect[0, 2]]

            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            m1 = rect[0, :2].expand_as(centers) < centers
            m2 = centers < rect[0, 2:].expand_as(centers)
            mask = (m1 + m2).gt(1)  # equivalent to logical-and

            boxes = boxes[mask].copy()
            classes = labels[mask]
            boxes[:, :2] = torch.max(
                boxes[:, :2], rect[:, :2].expand_as(boxes))
            boxes[:, :2] -= rect[:, :2].expand_as(boxes)
            boxes[:, 2:] = torch.min(
                boxes[:, 2:], rect[:, 2:].expand_as(boxes))
            boxes[:, 2:] -= rect[:, 2:].expand_as(boxes)

            return img, boxes, classes


def random_sample(img, boxes, labels):
    """Randomly sample the image by 1 of:
        - using entire original input image
        - sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
        - randomly sample a patch
    sample patch size is [0.1, 1] * size of original
    aspect ratio b/t .5 & 2
    keep overlap with gt box IF center in sampled patch

    Arguments:
        img (Image): the image being input during training

    Return:
        (img, boxes, classes)
            img (Image): the randomly sampled image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    sample_options = (
        # using entire original input image
        None,
        # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
        (0.1, None),
        (0.3, None),
        (0.7, None),
        (0.9, None),
        # randomly sample a patch
        (None, None),
    )

    mode = random.choice(sample_options)

    img, boxes, labels = crop(img, boxes, labels, mode)

    return img, boxes, labels


def rgb_to_hsv(r, g, b):
    """HSV: Hue, Saturation, Value
    source: https://github.com/python/cpython/blob/3.6/Lib/colorsys.py

    Converts an RGB color representation to HSV representation

    Arguments:
        (r, g, b)
        r (int): red channel (0-255)
        g (int): green channel (0-255)
        b (int): blue channel (0-255)
    Return:
        (h, s, v)
        h (int): position in the spectrum (0-300)
        s (int): color saturation ("purity") (0-100)
        v (int): color brightness (0-100)
    """
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc - minc) / maxc
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h / 6.0) % 1.0
    return h, s, v


def photometric_distort(img, boxes, classes):
    """photo-metric distortions
    https://arxiv.org/pdf/1312.5402.pdf

    -> manipulate contrast, brightness, color
    -> add random lighting noise
    """

    def convert(img, alpha=1, delta=0):
        """performs the distortion
        Arguments:
            img (Image): input image to be distorted
            alpha (float): probability of enhanchement (0.5-1.5, 1 leaves unchanged)
            delta (float): value of the distortion
        """

        tmp = img.float() * alpha + delta
        img = tmp.clamp_(min=0, max=255)

    img = img.copy()

    # brightness
    # prob: 0
    # value: [0, 255]. Recommend 32.
    if random.randrange(2):
        convert(img, delta=random.uniform(-32, 32))

    # contrast
    # prob: 0
    # value: [.5, 1.5]
    if random.randrange(2):
        convert(img, alpha=random.uniform(0.5, 1.5))

    # convert to HSV
    # img = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2HSV)
    img = img.convert('HSV')

    # saturation
    # prob: 0
    # [0.5, 1.5]
    if random.randrange(2):
        convert(img[:, :, 1], alpha=random.uniform(0.5, 1.5))

    # hue
    # prob: 0
    # value: [0,180]. Recommend 36
    if random.randrange(2):
        tmp = img[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        img[:, :, 0] = tmp

    # convert to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # random order img channels
    # prob: 0.0
    if random.randrange(2):
        # shuffle channels
        print()

    return img, boxes, classes


class TrainTransform(object):
    """Takes an input image and its annotation and transforms the image to
    help make the model more robust to various input object sizes and shapes
    during training.

    sample -> resize -> flip -> photometric (?)

    TODO: complete all steps of augmentation

    Return:
        transform (transform): the transformation to be applied to the the
        image
    """

    def __call__(self, img, anno):
        """
        Arguments:
            img (Image): the image being input during training
            anno (list): a list containing lists of bounding boxes
                (output of the target_transform) [bbox coords, class name]
        Returns:
            tuple of Tensors (image, anno)
        """

        # anno [[xmin, ymin, xmax, ymax, label_ind], ... ]
        anno = torch.Tensor(anno)
        boxes, labels = torch.split(anno, 4, 1)

        # SAMPLE - Random sample the image
        img, boxes, labels = random_sample(img, boxes, labels)

        # apply photo-metric distortions
        img, boxes, labels = photometric_distort(img, boxes, labels)

        # RESIZE to fixed size
        # resize = transforms.RandomSizedCrop(224)

        transforms.Compose([
            # sample,
            # resize,
            transforms.RandomHorizontalFlip()
            # photmetric
        ])
        return img


class SwapChannel(object):
    """Transforms a tensorized image by swapping the channels as specified in the swap

    modifies the input tensor

    Arguments:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Arguments:
            image (Tensor): image tensor to be transformed
        Returns:
            a tensor with channels swapped according to swap
        """
        temp = image.clone()
        for i in range(3):
            temp[i] = image[self.swaps[i]]
        return temp


def base_transform(dim, mean_values):
    """Defines the transformations that should be applied to test PIL Image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        dim (int): input dimension to SSD
        mean_values ( (int,int,int) ): average RGB of the dataset
            (104,117,123)

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    swap = (2, 1, 0)

    return transforms.Compose([
        transforms.Scale(dim),
        transforms.CenterCrop(dim),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        SwapChannel(swap),
        transforms.Normalize(mean_values, (1, 1, 1))
    ])
