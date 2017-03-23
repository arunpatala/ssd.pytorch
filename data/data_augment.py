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
import numpy as np
from PIL import Image
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
            left = random.randrange(width - w)
            top = random.randrange(height - h)
            rect = torch.LongTensor([[left, top, left+w, top+h]])


            overlap = jaccard(boxes, rect)
            if overlap.min() < min_iou and max_iou < overlap.max():
                continue
            t = transforms.ToTensor()
            p = transforms.ToPILImage()
            img = p(t(img)[:,rect[0, 1]:rect[0, 3], rect[0, 0]:rect[0, 2]])

            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            m1 = (rect[0, :2].expand_as(centers).lt(centers)).sum(1).gt(1)
            m2 = (centers.lt(rect[0, 2:].expand_as(centers))).sum(1).gt(1)


            mask = (m1 + m2).gt(1)  # equivalent to logical-and
            boxes = boxes[mask.expand_as(boxes)].copy()
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


def photometric_distort(image, boxes, classes):
    """photo-metric distortions
    https://arxiv.org/pdf/1312.5402.pdf

    -> manipulate contrast, brightness, color
    -> add random lighting noise
    """

    def convert(pix, prob=1, delta=0):
        """performs the distortion

        Args:
            pix (np): input image to be distorted
            prob (float): probability of enhanchement (0.5-1.5, 1 leaves unchanged)
            delta (float): value of the distortion
        """
        tmp = pix.astype(float) * prob + delta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        pix[:] = tmp

    def rand_brightness(pix, delta=32):
        """brightness
        prob: 0
        value: [0, 255]. Recommend 32.

        Args:
            pix (np): input image to be distorted
            delta(int): delta of the distortion
                (default: 32) [0, 255]
        """
        assert delta >= 0, "brightness delta must be non-negative."
        if random.randrange(2):
            convert(pix, delta=random.uniform(-delta, delta))

    def rand_contrast(pix, lower=0.5, upper=1.5):
        """contrast
        prob: 0
        value: [.5, 1.5]

        Args:
            pix (np): input image to be distorted
            lower (float): lower contrast bound
                (default: 0.5)
            upper (float): upper contrast bound
                (default: 1.5)
        """
        assert upper >= lower, "contrast upper must be >= lower."
        assert lower >= 0, "contrast lower must be non-negative."
        if random.randrange(2):
            convert(pix, prob=random.uniform(lower, upper))

    def convert_to_HSV(img, pix):
        """Converts from RGB to HSV

        Hue, Saturation, Value

        Args:
            img (Image): image
            pix (np): image
        """
        img = Image.fromarray(pix)
        img = img.convert('HSV')
        pix = np.array(img)

    def rand_saturation(pix, lower=0.5, upper=1.5):
        """saturation
        prob: 0
        [0.5, 1.5]

        Args:
            pix (np): input image to be distorted
            lower (float): lower saturation bound
                (default: 0.5)
            upper (float): upper saturation bound
                (default: 1.5)
        """
        assert upper >= lower, "saturation upper must be >= lower."
        assert lower >= 0, "saturation lower must be non-negative."
        if random.randrange(2):
            convert(pix[:, :, 1], prob=random.uniform(lower, upper))

    def rand_hue(pix, delta=36):
        """hue
        prob: 0
        value: [0,180]. Recommend 36

        Args:
            pix (np): input image to be distorted
            delta(int): delta of distortion
                (default: 36) [0,180]
        """
        assert delta >= 0, "hue delta must be non-negative."
        if random.randrange(2):
            tmp = pix[:, :, 0].astype(int) + \
                random.randint(-delta / 2, delta / 2)
            tmp %= 180
            pix[:, :, 0] = tmp

    def convert_to_RGB(img, pix):
        """Converts from HSV to RGB

        Args:
            img (Image): image
            pix (np): image
        """
        img = Image.fromarray(pix)
        img = img.convert('RGB')
        pix = np.array(img)

    def rand_lighting_noise(pix):
        """random order img channels to add random lighting noise
        prob: 0.0

        Args:
            pix (np): input image to be distorted
        """
        channel_perms = ((0, 1, 2), (0, 2, 1),
                         (1, 0, 2), (1, 2, 0),
                         (2, 0, 1), (2, 1, 0))
        if random.randrange(2):
            swap = random.choice(channel_perms)
            # shuffle channels
            shuffle = SwapChannel(swap)
            tns = torch.from_numpy(pix)
            tns = shuffle(tns)
            pix = tns.numpy()

    def apply_distorts(img):
        """Randomly applies all of the distortions defined above

        Args:
            img (Image): input image to be distorted
        """

        img = img.copy()     # PIL
        pix = np.array(img)  # NP

        rand_brightness(pix)
        if random.randrange(2):
            rand_contrast(pix)
            convert_to_HSV(img, pix)
            rand_saturation(pix)
            rand_hue(pix)
            convert_to_RGB(img, pix)
        else:
            convert_to_HSV(img, pix)
            rand_saturation(pix)
            rand_hue(pix)
            convert_to_RGB(img, pix)
            rand_contrast(pix)
        rand_lighting_noise(pix)
        return Image.fromarray(pix)

    return apply_distorts(image), boxes, classes


class TrainTransform(object):
    """Takes an input image and its annotation and transforms the image to
    help make the model more robust to various input object sizes and shapes
    during training.

    sample -> photometric -> resize -> flip

    TODO: complete all steps of augmentation

    Return:
        transform (transform): the transformation to be applied to the the
        image
    """

    def __call__(self, img, anno):
        """
        Args:
            img (Image): the image being input during training
            anno (list): a list containing lists of bounding boxes
                (output of the target_transform) [bbox coords, class name]
        Return:
            tuple of Tensors (image, anno)
        """

        # anno [[xmin, ymin, xmax, ymax, label_ind], ... ]
        anno = torch.Tensor(anno)
        boxes, labels = torch.split(anno, 4, 1)

        # SAMPLE - Randomly sample a crop of image
        img, boxes, labels = random_sample(img, boxes, labels)

        # DISTORT - apply photo-metric distortions
        img, boxes, labels = photometric_distort(img, boxes, labels)

        # RESIZE to fixed size
        # resize = transforms.RandomSizedCrop(224)

        # FLIP

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

    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
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

    Args:
        dim (int): input dimension to SSD
        mean_values ( (int,int,int) ): average RGB of the dataset
            (104,117,123)

    Return:
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
