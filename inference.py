import torch
from PIL import Image
import numpy as np
import logging
import cv2
from pspnet import PSPNet
import matplotlib.pyplot as plt
import time
DEVICE = 'cuda:0'
models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}
class SegmentMetrics(object):
    def __init__(self, truth, pred):
        """
        truh : np_array (W , H) values in classify form 0,1,...
        pred : np_array (W , H) values in classify form 0,1,...
        """
        self.truth = truth
        self.pred = pred
        self.smooth = 0.001
    def intersection(self):
        return np.sum(np.logical_and(self.pred, self.truth))
    def union(self):
        return np.sum(np.logical_or(self.pred, self.truth))
    def IOU(self):
        return (self.intersection() + self.smooth) / (self.union() + self.smooth)
    def mIOU(self):
        return np.mean(self.IOU())

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    #net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda(0)
    return net, epoch
net,_ = build_network(None, backend='resnet18')

net.load_state_dict(torch.load('./67Loss_0.037000544369220734.pth'))

image=cv2.imread("./data1./data_24.jpg")


def tensorShow(tensor):
    print("shape",tensor.shape)
    img = tensor.permute(1,2,0).cpu()#1,2,0
    img = img.detach().numpy()
    
    cv2.imshow("predicted", img[:,:,0]*255)
    cv2.waitKey(0)

colormap = [[0,0,0],[106, 61, 154],[227, 26, 28],[31, 120, 180]]

cm = np.array(colormap).astype('uint8')
def predict(img):
    img = torch.from_numpy(img).permute(2,0,1).to(DEVICE).unsqueeze(0).float()
    
    img = img.cuda(0)   
    out, _ = net(img)
    print(out)
    pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
    pre_label = np.asarray(pre_label, dtype=np.uint8)
    pre = cm[pre_label]
    
    pre1 = Image.fromarray(pre.astype("uint8"), mode='RGB')
    print()
    # metric=SegmentMetrics(pre_label,out)
    #print(metric.mIOU())
    pre1.save( "ac" + '.png')
    print("Done")



    # print("tensor",x_tensor.shape)
    # mask, _ =net(x_tensor)
    # print('out put',mask.shape)


    # return mask
t=time.time()
predict(image)
print((time.time()-t))
