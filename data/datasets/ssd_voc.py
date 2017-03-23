"""Custom VOC Dataset wrapper for SSD

Ellis Brown
"""

import sys
from PIL import Image

from . import VOCDetection
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

class SSDVOC(VOCDetection):
    """Wrapper for VOCDetection dataset that overrides the getitem function
    for training so that the data augmentation transforms can access the
    annotation
    """

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = Image.open(self._imgpath % img_id).convert('RGB')
        width, height = img.size

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if self.image_set == 'train':
                img = self.transform(img, target)
            else:
                img = self.transform(img)
        img.squeeze_(0)

        return img, target
