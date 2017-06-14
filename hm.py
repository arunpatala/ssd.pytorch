from torchsample.callbacks import Callback
import torch
from polarbear import *
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import gpustat
import gc


class PosNeg(Callback):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self, net, ct, val):
        self.net = net
        self.ct = ct
        self.softmax = nn.Softmax()
        self.val = val

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        #val_detect(self.net, self.ct, self.val)
        pass

    def on_epoch_end(self, epoch, logs=None):
        #val_detect(self.net, self.ct, self.val)
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end2(self, batch, logs=None):
        gpustat.print_gpustat(json=False)
    
    def on_batch_end(self, batch, logs=None):
        fpath = "weights/logs/{batch}".format(batch=batch)
        fpath = fpath + '{type}.jpg'
        img = img2Image(self.net.imgs.data)
        himg = self.ct.himg
        himg.plot().save(fpath.format(type="himg"))
        pnimg = self.ct.pnimg
        AnnImg(img, pnimg.ann).plotc().save(fpath.format(type="pnimg"))
        torch.save(himg,fpath.format(type="th"))
        
    def on_train_begin(self, logs=None):
        #val_detect(self.net, self.val)
        pass

    def on_train_end(self, logs=None):

        pass

def val_detect(model, ct, val_dataset, cuda=True):
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    model.eval()
    model.phase = "test"
    #help(tqdm)
    loss = 0
    pdc,fpc,fnc = 0,0,0
    s = set()
    for  i,(img, ann_gt) in tqdm(enumerate(iter(val_loader)), total=len(val_loader)):
        if cuda: img = img.cuda()
        dets = model.forward(Variable(img))
        hmsave(model.conf_output,i,folder='weights/logsv/',fmap=model.fmap)

        #loss += ct.forward(lcp,Variable(ann_gt)).data[0]
        #print(dets)
        dim = img.size(2)
        ann = Ann(tensor=dets[0].data.cpu(),dim=img.size(2))
        s.update(ann.unique())
        #print(ann)
        if ann.dets is None: continue
        if ann.fconf(50).count == 0: continue
        #ann.cl -= 1
        img = img2Image(img)
        ann = ann.fconf(50)
        aimg = AnnImg(img, ann)
        AnnImg(img, ann).plot(save="weights/logsv/{i}a.jpg".format(i=i))
        ann = ann.allNMS(0.5)
        aimg = AnnImg(img, ann)
        #print(Ann,ann_gt)
        agt = Ann(dets=ann_gt[0].numpy(), dim=dim)
        #print(agt)
        #print(ann)
        p,r,(gt,pd,fn,fp) = agt.prec_recall(ann)
        if pd.nelement()>0:
            pdc += pd.size(0)
        if fp.nelement()>0:
            fpc += fp.size(0)
        if fn.nelement()>0:    
            fnc += fn.size(0)
        #print(p,r,gt.shape, pd.shape, fp.shape, fn.shape)
        AnnImg(img, Ann(dets=pd.numpy())).plot(save="weights/logsv/{i}pd.jpg".format(i=i))
        AnnImg(img, Ann(dets=fp.numpy())).plot(save="weights/logsv/{i}fp.jpg".format(i=i))
        AnnImg(img, Ann(dets=fn.numpy())).plot(save="weights/logsv/{i}fn.jpg".format(i=i))
        AnnImg(img, ann).plot(save="weights/logsv/{i}.jpg".format(i=i))
        AnnImg(img, ann).plot(rect=False, label=False, save="weights/logsv/{i}c.jpg".format(i=i))
        #print(s)
    print("pdc", "fpc", "fnc")
    print(pdc, fpc, fnc)
    print("prec", pdc/(pdc+fpc+1), "recall", pdc/(pdc+fnc+1))
    model.phase = "train"
    model.train()

def img2Image(img):
    _,C,H,W, = img.size()
    rgb_means = (104, 117, 123)
    new_img = img[0].cpu().clone().numpy().transpose((1,2,0))
    new_img += rgb_means
    return Image.fromarray(new_img.astype('uint8'))
