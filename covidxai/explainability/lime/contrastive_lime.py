# something about torch using pointers.. need to clone()
#%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from skimage.color import rgb2grey, gray2rgb
from skimage.segmentation import mark_boundaries

import os
import sys
sys.path.insert(0, '/home/csgrads/sfegh001/codes')
import importlib
import clime_utils
from clime_utils import *

import lime
from lime import lime_image

import torch
import torch.optim as optim
import sys
import argparse
import os
import shutil
import time
import datetime
import cv2
from PIL import Image
import numpy as np
from skimage import data, img_as_float
#import pytorch_ssim
import seaborn as sns

import matplotlib
import math
import numpy as np
from PIL import Image
import os
import random
from scipy import misc
from scipy.ndimage import filters
import importlib

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

torch.cuda.set_device(1)

import torchvision.models as models
import importlib

sys.path.insert(0, '/home/sfegh001/amir/pytorch/alexmodel/')
sys.path.insert(0, '/home/csgrads/sfegh001/codes')
#ssim_loss = pytorch_ssim.SSIM()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5), nn.MaxPool2d((2,2)), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(10, 20, kernel_size=5), nn.MaxPool2d((2,2)), nn.ReLU())
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = self.conv1(x)#F.relu(F.max_pool2d(self.conv1(x), 2))
        #print(x.size())
        x = self.conv2(x)#F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.size())
        x = x.view(-1, 3380)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        SoftMax = nn.Softmax()
        return F.log_softmax(x, dim=1)


def pred_func1(tens):
    vx = Variable(tens.data.clone().cuda())
    yhat = model(vx)
    yhat = SoftMax(yhat)
    return yhat.data.cpu().numpy()


def pred_func(np_img):
    # np_img is (batch=10, channel=3, height, width) when this
    # function is called in lime_image during sampling
    # change to (batch=10, channel=1, height, width)
    # don't need transform since data is in format we want
    # np_img and tmp share the same memory
    if len(np_img.shape) > 2:
        np_img = rgb2grey(np_img)
        np_img = np.expand_dims(np_img, 1)
    else:
        np_img = np.expand_dims(np_img, 0)
        np_img = np.expand_dims(np_img, 0)
    tmp = torch.from_numpy(np_img)
    x = tmp.clone()
    vx = Variable(x.cuda(), requires_grad=True).float()
    yhat = model(vx)
    # network uses log softmax
    # soft = nn.Softmax(dim=1)
    # yhat = soft(yhat)
    # yhat = torch.exp(yhat)
    yhat = SoftMax(yhat)
    return yhat.data.cpu().numpy()


model = Net()
model_state = torch.load('/home/csgrads/sfegh001/codes/mnist_8_3x3_white_block/mnist_trained_2018_09_16_08_11PM_biased_8.pkl')
#model_state = torch.load('/home/csgrads/sfegh001/codes/mnist_trained_2018_08_21_08_00PM.pkl')
model.load_state_dict(model_state)
model = model.cuda()
model.eval()
D = model

use_cuda = True
batch_size = 1
mean = [0.5]
std = [0.5]
SoftMax = nn.Softmax(dim=1)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                           transforms.Resize((64, 64)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=mean,std=std)
                       ])),
        batch_size=batch_size, shuffle=False)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((64,64)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=1, shuffle=False)



lbl_st = torch.load('../lbl_mnist.pkl')
each_class_acc = lbl_st['each_class_acc']
cnts = lbl_st['cnts']
example_numbers = lbl_st['example_numbers']


print('[ ', end='')
for i in range(10):
    for j in range(5):
        print( i, ', ', example_numbers[i][j], end='')



explainer = lime_image.LimeImageExplainer()
num_features=5
fontsize=8
color_out = "grey"
file_suffix = ".jpg" if color_out == "rgb" else "_grey.jpg"

num_samples_list = [ 12000, 500, 1000, 5000, 25000]

num_samples = 5000
dest_path = "biased_8/lime_results/"
save_dir = dest_path

save_dir = os.path.join(dest_path, "samples_" + str(num_samples))
indices_to_test_biased = [ 2185 , 716 , 956 , 421]# , 583 , 18 , 938 , 447 , 969 , 211 , 478 , 2995 , 3550 , 175 , 1206 , 479 , 1559]
indices_to_test_true = [324 ,  552 ,  1898 ,  2329 ,  2631 ,  96 ,  956 ,  1868 ,  1909 ,  2343 ,  326 ,  421 ,  492 ,
                        613 ,  646 ,  18 ,  883 ,  1128 ,  1611 ,  2770 ,  115 ,  497 ,  707 ,  1263 ,  1453 ,  4360 ,
                        4548 ,  4763 ,  5982 ,  6598 ,  457 ,  508 ,  870 ,  1436 ,  1587 ,  810 ,  1216 ,  1500 ,
                        1520 ,  1721 ,  184 ,  266 ,  495 ,  542 ,  591 ,  151 ,  359 ,  448 ,  479 ,  1553]

#my_indices = [448, 184, 1721, 1587, 552 ]
#my_indices = [552, 18, 497, 4548]
my_indices = [421, 6632]
'''
for i in indices_to_test_true:
    print(i)
    vinput1,target = get_input(train_loader, i)
    save_image(vinput1, 'samples/' + str(i) + '.jpg')
'''

lst_vals = list()
lst_vals.append(list())
lst_vals.append(list())


class ContrastiveLIME:
    """
    Contrastive Local Interpretable Model-agnostic Explanations.
    """
    def __init__(self, model):
        self.model = model


weights21= [i / 200 for i in range(1, 200)]
weights22= [1 + i / 200 for i in range(1, 200)]
#weights22 = [-1 - (i / 140) for i in range(1, 140)]
#weights2 = [-i / 140 for i in range(1, 140)]
weights2 =   weights21  + weights22

weights1 = [1- i/100 for i in range(1,60)]
weights11 = [1 + i/100 for i in range(1,60)]
weights1 = weights1 + weights11
weights1 = [1]
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.22)
for ii, tindex in enumerate(my_indices):
    #if ii < 1:
    #    continue
    vinput1, label = get_input(test_loader, tindex)
    pic_name = "index_" + str(tindex) + "_label_" + str(label.cpu().data.numpy().squeeze()) + "_samples_" + str(num_samples)
    pic_dest = os.path.join(save_dir, pic_name + file_suffix)
    pic_dest_simple = os.path.join(save_dir, pic_name + '_simple_' + file_suffix)
    print(pic_dest)
    image = vinput1
    np_image = image.cpu()[0][0].numpy()
    #probs = pred_func1(image).squeeze()
    lbl = 8  # predictions[1]
    probs = pred_func(np_image).squeeze()
    aa = probs[label.item()]
    bb = probs[lbl]
    predictions = probs.argsort()[::-1]
    pred_probs = np.sort(probs)[::-1]
    pred_probs[0] = aa
    pred_probs[1] = bb
    exptop2 = explainer.explain_instance(np_image, pred_func,
                                                                 hide_color=True, num_samples=num_samples,
                                                                 top_labels=10
                                         , segmentation_fn=segmenter
                                         )
    a,b  = exptop2.get_image_and_mask(label.item(),positive_only=True,num_features=100000, hide_rest=True)
    b = b.astype('float')
    a = (a - a.min()) / (a.max() - a.min() + 0.000001)
    c,d  = exptop2.get_image_and_mask(lbl,positive_only=True,num_features=100000, hide_rest=True)
    c = (c - c.min()) / (c.max() - c.min() + 0.000001)
    d = d.astype('float')
    #a = (a - a.min()) / a.max()
    #b = (b - b.min()) / b.max()
    #c = (c - c.min()) / c.max()
    # = (d - d.min()) / d.max()
    plt.close()
    '''
    fig = plt.figure()
    fig.add_subplot(2,5,1)
    plt.imshow(a)
    plt.title('a')
    fig.add_subplot(2,5,2)
    plt.imshow(b)
    plt.title('b')
    fig.add_subplot(2,5,3)
    plt.imshow(c)
    plt.title('c')
    fig.add_subplot(2,5,4)
    plt.imshow(d)
    plt.title('d')
    fig.add_subplot(2,5,5)
    plt.imshow(a-c)
    plt.title('a-c')
    fig.add_subplot(2,5,6)
    plt.imshow(b-d)
    plt.title('b-d')
    fig.add_subplot(2,5,7)
    plt.imshow(a + -0.05 * c)
    plt.title('explanation')
    fig.add_subplot(2,5,8)
    plt.imshow(0.8 * a + 0.05 * c)
    plt.title('0.8 * a + 0.05 * c')
    '''
    #weights2 = [1, -1, 2, -2, 3, -3, 4, -4, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1, 0.05, -0.05, 0.01, -0.01, 0.001, -0.001, -0.3, -0.25, -0.15]
    print(label.item(), ' , ', lbl)
    print(pred_probs[0], ' , ', pred_probs[1])
    ch1 = 0
    ch2 = 0
    avg_pic = Tensor(64,64,3).fill_(0)
    wcnt = 0
    for we1 in weights1:
        for we2 in weights2:
            #apc = rgb2grey(we1 * a + we2 *c)
            apc = -we1 * a[:,:,0] + we2 * c[:,:,0]
            apc = Tensor(apc)
            #avg_pic += apc.clone()
            apc = apc.unsqueeze(0).unsqueeze(0)
            apc = make_in_range(apc + vinput1.clone(), vinput1.min(), vinput1.max())
            yapc = D(apc)
            val1 = SoftMax(yapc)[0, label.item()].item()
            val2 = SoftMax(yapc)[0, lbl].item()
            #break
            if pred_probs[0] - val1 < 0.03 or val2 - pred_probs[1] < 0.03:# or val1 < 0.15 or val2 < 0.15:
                continue
            avg_pic += Tensor(we1 * -a + we2 * c)
            ch1 += val1 - pred_probs[0]
            ch2 += val2 - pred_probs[1]
            wcnt += 1
            #print('w1: ', we1, ', we2: ', we2, ', new: ', val1 , ' , ', val2)
    if wcnt == 0:
        print('ridi')
        continue
    else:
        print('idx: ', tindex, ', wcnt: ', wcnt)
    #print('changes: ', ch1 /wcnt, ',', ch2 /wcnt )
    lst_vals[0].append(ch1/wcnt)
    lst_vals[1].append(ch2 / wcnt)
    plt.close()
    fig = plt.figure()
    ee = avg_pic/ wcnt
    #ee = make_in_range(ee, 0, 1)
    ee = ee[:,:,0].clone() #+ vinput1.clone()
    print('ee max and min: ', ee.max().item(), ', ', ee.min().item())
    rev_inp = vinput1.clone() *0.5 + 0.5
    rev_inp = rev_inp.squeeze(0).squeeze(0)
    eec = make_in_range(ee, vinput1.min().item(), vinput1.max().item())
    eec = ee
    exp_img = get_overlayed_image(rev_inp.detach().cpu().numpy(),  eec.detach().cpu().numpy())
    plt.imshow(exp_img)
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    #plt.savefig('abcd.jpg', bbox_inches='tight',transparent=True, pad_inches=0)
    plt.savefig(pic_dest, bbox_inches='tight',transparent=True, pad_inches=0)
    plt.close()
    apc = 1 * a[:,:,0] + 1 * c[:,:,0]
    apc = Tensor(apc)
    # avg_pic += apc.clone()
    #apc = apc.unsqueeze(0).unsqueeze(0)
    #apc = make_in_range(apc, vinput1.min(), vinput1.max())
    #exp_img = get_overlayed_image(rev_inp.detach().cpu().numpy(),
    #                                  apc.detach().cpu().numpy())
    #plt.imshow(exp_img)
    #plt.savefig(pic_dest_simple)
    #ee = rgb2grey(3 *c)
    #qq2 = Tensor(ee)
    #qq2 = qq2.unsqueeze(0).unsqueeze(0)
    #yy2 = D(qq2)
    #print(pred_probs[0], ' , ' , pred_probs[1])
    #print('new: ', SoftMax(y2)[0,label.item()].item(), ' , ' , SoftMax(y2)[0,lbl].item())
    #print('new: ', SoftMax(yy2)[0,label.item()].item(), ' , ' , SoftMax(yy2)[0,lbl].item())
    #plt.show()
    #break


torch.save(lst_vals, 'change_vals.pth')
print(np.mean(lst_vals[0]))
print(np.mean(lst_vals[1]))

for a,b in zip(lst_vals[0], lst_vals[1]):
    print(a, ' , ', b)