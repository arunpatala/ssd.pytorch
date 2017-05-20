from torchsample.callbacks import Callback
import torch
from polarbear import *
from PIL import Image

class PosNeg(Callback):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self, net, ct):
        self.net = net
        self.ct = ct

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pos = self.ct.pos_boxes
        neg = self.ct.neg_boxes
        img = self.net.imgs
        targets = self.ct.targets
        torch.save([img, targets, pos, neg], "tmp.th")
        save(img, targets, pos, neg, batch)

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

def save(img, tgts, pos, neg, batch):
    assert(img.size(0)==1)
    rgb_means = (104, 117, 123)
    new_img = img[0].data.clone().numpy().transpose((1,2,0))
    new_img += rgb_means
    img = Image.fromarray(new_img.astype('uint8'))
    pann = Ann(dets=(pos*300).round().int().numpy())
    nann = Ann(dets=(neg*300).round().int().numpy())
    tann = Ann(dets=(tgts.data[0][:,:-1]*300).int().numpy())
    rect = True
    pimgs = pann.plot_size(img, color=(0,255,0))
    for i in range(len(pimgs)):
        pimgs[i].save("tmp/"+str(batch)+"_"+str(i)+"pimg.jpg")
    nimgs = nann.plot_size(img, color=(255,0,0))
    for i in range(len(nimgs)):
        nimgs[i].save("tmp/"+str(batch)+"_"+str(i)+"nimg.jpg")

    pimg = AnnImg(img, pann).plot(label=False,color=(0,255,0),rect=rect).img
    pimg.save("tmp/"+str(batch)+"pimg.jpg")
    pnimg = AnnImg(pimg, nann).plot(label=False,color=(255,255,0), rect=rect).img
    pnimg.save("tmp/"+str(batch)+"pnimg.jpg")
    tpnimg = AnnImg(pnimg, tann).plot(label=False,color=(0,255,255), rect=rect).img
    tpnimg.save("tmp/"+str(batch)+"tpnimg.jpg")
    AnnImg(img, nann).plot(label=False, color=(255,255,0), rect=rect, save = "tmp/"+str(batch)+"nimg.jpg")
    timg = AnnImg(img, tann).plot(label=False,color=(0,255,255), rect=rect).img
    timg.save("tmp/"+str(batch)+"timg.jpg")
    rect = False
    pimg = AnnImg(img, pann).plot(label=False,color=(0,255,0),rect=rect).img
    pimg.save("tmp/"+str(batch)+"pimgc.jpg")
    pnimg = AnnImg(pimg, nann).plot(label=False,color=(255,255,0), rect=rect).img
    pnimg.save("tmp/"+str(batch)+"pnimgc.jpg")
    tpnimg = AnnImg(pnimg, tann).plot(label=False,color=(0,255,255), rect=rect).img
    tpnimg.save("tmp/"+str(batch)+"tpnimgc.jpg")
    nimg = AnnImg(img, nann).plot(label=False,color=(255,255,0), rect=rect).img
    nimg.save("tmp/"+str(batch)+"nimgc.jpg")
    timg = AnnImg(img, tann).plot(label=False,color=(0,255,255), rect=rect).img
    timg.save("tmp/"+str(batch)+"timgc.jpg")


