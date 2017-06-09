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
        val_detect(self.net, self.ct, self.val)
        #pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end2(self, batch, logs=None):
        gpustat.print_gpustat(json=False)

    def on_batch_end(self, batch, logs=None):
        pos = self.ct.pos_boxes
        neg = self.ct.neg_boxes
        img = self.net.imgs
        conf = self.softmax((self.ct.conf_data.view(-1, 6)))
        hmsave(conf, batch)
        targets = self.ct.targets
        #torch.save([img, targets, pos, neg], "tmp.th")
        save(img, targets, pos, neg, batch)
        #gpustat.print_gpustat(json=False)

    def on_train_begin(self, logs=None):
        #val_detect(self.net, self.val)
        pass

    def on_train_end(self, logs=None):

        pass
def hmsave(conf, batch):
    print("conf", conf.size())
    i = 75
    heatmap1 = (conf.data[:i*i,0]*255).int()
    heatmap1 = heatmap1.contiguous().view((i,i)).cpu().numpy()
    Image.fromarray(heatmap1.astype('uint8')).save("weights/logs/"+str(batch)+"_himg.jpg")
    heatmap2 = (conf.data[i*i:,0]*255).int()
    i = 37
    heatmap2 = heatmap2.contiguous().view((37,37)).cpu().numpy()
    print(heatmap2)
    Image.fromarray(heatmap2.astype('uint8')).save("weights/logs/"+str(batch)+"_hhimg.jpg")
    
def save(img, tgts, pos, neg, batch):
    assert(img.size(0)==1)
    _,C,H,W, = img.size()
    """rgb_means = (104, 117, 123)
    new_img = img[0].data.cpu().clone().numpy().transpose((1,2,0))
    new_img += rgb_means
    img = Image.fromarray(new_img.astype('uint8'))"""
    img = img2Image(img.data)


    pann = Ann(dets=(pos*H).round().int().cpu().numpy())
    #print("pos",pann.unique())
    nann = Ann(dets=(neg*H).round().int().cpu().numpy())
    #print("neg",nann.unique())
    tann = Ann(dets=(tgts.data[0][:,:-1]*H).int().cpu().numpy())
    #print("truth",tann.unique())
    rect = True
    """
    pimgs = pann.plot_size(img, color=(0,255,0), rect=False)
    for i in range(len(pimgs)):
        pimgs[i].save("weights/logs/"+str(batch)+"_"+str(i)+"pimg.jpg")
    nimgs = nann.plot_size(img, color=(255,0,0), rect=False)
    for i in range(len(nimgs)):
        nimgs[i].save("weights/logs/"+str(batch)+"_"+str(i)+"nimg.jpg")
    """
    pimg = AnnImg(img, pann).plot(label=False,color=(0,255,0),rect=rect).img
    #pimg.save("weights/logs/"+str(batch)+"pimg.jpg")
    pnimg = AnnImg(pimg, nann).plot(label=False,color=(255,0,0), rect=rect).img
    pnimg.save("weights/logs/"+str(batch)+"pnimg.jpg")
    tpnimg = AnnImg(pnimg, tann).plot(label=False,color=(0,255,255), rect=rect).img
    #tpnimg.save("weights/logs/"+str(batch)+"tpnimg.jpg")
    AnnImg(img, nann).plot(label=False, color=(255,0,0), rect=rect, save = "weights/logs/"+str(batch)+"nimg.jpg")
    timg = AnnImg(img, tann).plot(label=False,color=(0,255,255), rect=rect).img
    timg.save("weights/logs/"+str(batch)+"timg.jpg")
    rect = False
    nimg = AnnImg(img, nann).plot(label=False,color=(255,0,0),rect=rect).img
    #pimg.save("weights/logs/"+str(batch)+"pimgc.jpg")
    pnimg = AnnImg(nimg, pann).plot(label=False,color=(0,255,0), rect=rect).img
    pnimg.save("weights/logs/"+str(batch)+"pnimgc.jpg")
    tpnimg = AnnImg(pnimg, tann).plot(label=False,color=(0,255,255), rect=rect).img
    #tpnimg.save("weights/logs/"+str(batch)+"tpnimgc.jpg")
    nimg = AnnImg(img, nann).plot(label=False,color=(255,0,0), rect=rect).img
    #nimg.save("weights/logs/"+str(batch)+"nimgc.jpg")
    timg = AnnImg(img, tann).plot(label=False,color=(0,255,255), rect=rect).img
    #timg.save("weights/logs/"+str(batch)+"timgc.jpg")

def val_detect(model, ct, val_dataset):
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    model.eval()
    model.phase = "test"
    #help(tqdm)
    loss = 0
    pdc,fpc,fnc = 0,0,0
    s = set()
    for  i,(img, ann_gt) in tqdm(enumerate(iter(val_loader)), total=len(val_loader)):
        dets = model.forward(Variable(img.cuda()))
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
    print("prec", pdc/(pdc+fpc), "recall", pdc/(pdc+fnc))
    model.phase = "train"
    model.train()

def img2Image(img):
    _,C,H,W, = img.size()
    rgb_means = (104, 117, 123)
    new_img = img[0].cpu().clone().numpy().transpose((1,2,0))
    new_img += rgb_means
    return Image.fromarray(new_img.astype('uint8'))