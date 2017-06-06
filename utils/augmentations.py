import torch
from torchvision import transforms
import cv2
import numpy as np
import types
import math
import random
from layers.box_utils import jaccard
from PIL import Image

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes, labels):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img)


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the np.ndarray, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std


class RandomSaturation(object):
    def __init__(self, lower=0.2, upper=4):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes, labels):
        if random.randrange(2):
            tmp = image[:, :, 1].astype(float) * \
                random.uniform(self.lower, self.upper)
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:, :, 1] = tmp
        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18):
        self.delta = delta

    def __call__(self, image, boxes, labels):
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + \
                random.randint(-self.delta, self.delta)
            tmp %= 180
            image[:, :, 0] = tmp
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes, labels):
        if random.randrange(2):
            swap = random.choice(self.perms)
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current
        self.transform = transform


    def __call__(self, image, boxes, labels):
        image = np.transpose(image,(1,2,0)).astype('uint8')
        #print(image.shape)
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        image = np.transpose(image,(2,0,1)).astype('float64')
        
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes, labels):
        if random.randrange(2):
            alpha = random.uniform(self.lower, self.upper)
            #print("image", len(image))
            #print("alpha", alpha)
            image *= alpha
            image[image < 0] = 0
            image[image > 255] = 255
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=64):
        self.delta = delta
        assert self.delta >= 0, "brightness delta must be non-negative."

    def __call__(self, image, boxes, labels):
        if random.randrange(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image[image < 0] = 0
            image[image > 255] = 255
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor):
        # may have to call cv2.cvtColor() to get to BGR
        return tensor.cpu().numpy().astype(np.float32)


class ToTensor(object):
    def __call__(self, cvimage):
        return torch.from_numpy(cvimage)


class RandomSampleCrop(object):
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
    def __init__(self, dim=600, fmin=1):
        self.dim = dim
        self.fmin = fmin

    def __call__(self, img, boxes, labels):
        #mode = random.choice(self.sample_options)
        boxes = torch.from_numpy(boxes).clone()
        labels = torch.from_numpy(labels).clone()
        _, height, width = img.shape
        assert(height==width)
        boxes *= height
        w = self.dim #random.randrange(int(0.3 * width), width)
        h = self.dim #random.randrange(int(0.3 * height), height)
        h = w
        #print("ww", width, w)
        while True:
            left = random.randrange(width - w)
            top = random.randrange(height - h)
            rect = torch.LongTensor([[left, top, left + w, top + h]])
            #overlap = jaccard(boxes, rect)
            #if overlap.min() < min_iou and max_iou < overlap.max():
            #    continue
            t = ToTensor()
            # p = transforms.ToPILImage()
            image = t(img)[:, rect[0, 1]:rect[0, 3], rect[0, 0]:rect[0, 2]]

            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            #print(rect)
            #print(centers)
            m1 = (rect[0, :2].float().expand_as(centers).lt(centers)).sum(1).gt(1)
            m2 = (centers.lt(rect[0, 2:].float().expand_as(centers))).sum(1).gt(1)
            mask = (m1 + m2).gt(1).squeeze().nonzero().squeeze()

            if mask.nelement() < self.fmin: 
                #print("continue")
                continue
            boxes = boxes[mask].clone()
            #print("boxes", boxes)
            classes = labels[mask]
            rect = rect.expand_as(boxes).float()
            #print("rect",rect)
            boxes[:, :2] = torch.max(boxes[:, :2], rect[:, :2])
            #print("boxes", boxes)
            boxes[:, :2] -= rect[:, :2]
            #print("boxes", boxes)
            boxes[:, 2:] = torch.min(boxes[:, 2:], rect[:, 2:])
            #print("boxes", boxes)
            boxes[:, 2:] -= rect[:, :2]
            #print("boxes", boxes)
            
            b = boxes.clone()
            #print("width", (b[:,2]-b[:,0]).min(), (b[:,2]-b[:,0]).max())
            #print("heights", (b[:,3]-b[:,1]).min(), (b[:,3]-b[:,1]).max())
            return image.numpy(), boxes.numpy()/self.dim, classes.numpy()


class Expand(object):
    def __call__(self, image, boxes, labels):
        
        if random.randrange(2):
            return image, boxes, labels

        depth ,height, width= image.shape
        boxes *= height

        ratio = random.uniform(1, 4)
        left = random.randint(0, int(width * ratio) - width)
        top = random.randint(0, int(height * ratio) - height)

        expand_image = np.empty(
            (depth, int(height * ratio), int(width * ratio)),
            dtype=image.dtype)
        expand_image[:, :, :] = 0
        expand_image[:, top:top + height, left:left + width] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (left, top)
        boxes[:, 2:] += (left, top)

        return image, boxes/image.shape[1], labels


class Zoom(object):

    def __init__(self, bmin=30, bmax=200, tile=900):
        self.bmin = bmin
        self.bmax = bmax
        self.tile = tile

    def __call__(self, image, boxes, labels):
        bmin, bmax = self.bmin, self.bmax

        depth, height, width = image.shape
        w = ((boxes[:,2]-boxes[:,0])*width).round().astype('int32')
        wmin, wmax = w.min(), w.max()
        self.min_zoom = bmin/wmin
        self.max_zoom = bmax/wmax

        h = ((boxes[:,3]-boxes[:,1])*height).astype('int32')
        hmin, hmax = h.min(), h.max()
        self.min_zoom = max(bmin/hmin, self.min_zoom)
        self.max_zoom = min(bmax/hmax, self.max_zoom)


        #print("h", self.min_zoom, self.max_zoom, min(hmin, wmin), max(hmax, wmax))

        ratio = random.uniform(self.min_zoom, self.max_zoom)
        w = int(width * ratio)
        w = max(w, self.tile+1)
        w = min(w, 2*self.tile)
        h = w

        img = Image.fromarray(np.transpose(image,(1,2,0)).astype('uint8'))
        img = img.resize((w,h))
        w = ((boxes[:,2]-boxes[:,0])*w).round().astype('int32')
        wmin, wmax = w.min(), w.max()

        h = ((boxes[:,3]-boxes[:,1])*h).astype('int32')
        hmin, hmax = h.min(), h.max()

        #print("hq", min(hmin, wmin), max(hmax, wmax))



        image = np.asarray(img, dtype = np.float)
        image = np.transpose(image, (2,0,1))

        return image, boxes, labels



class RandomMirror(object):
    def __call__(self, image, boxes, labels):
        _, width, _ = image.shape
        if random.randrange(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            #boxes[:, 0::2] = 1 - boxes[:, 2::-2]
            boxes[:, 1] = 1 - boxes[:, 1]
            boxes[:, 3] = 1 - boxes[:, 3]
        if random.randrange(2):
            image = image[:, :, ::-1]
            boxes = boxes.copy()
            #boxes[:, 0::2] = 1 - boxes[:, 2::-2]
            boxes[:, 0] = 1 - boxes[:, 0]
            boxes[:, 2] = 1 - boxes[:, 2]
        return image, boxes, labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps=(2,1,0)):
        self.swaps = swaps

    def __call__(self, image, boxes, labels):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image.transpose(*swaps)
        return image, boxes, labels
