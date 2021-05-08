import argparse
import os
import shutil
import time
import datetime
from PIL import Image
from skimage import color
import matplotlib.cm as cm
import torch
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
#import cv2
import torch.optim as optim

import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import pyplot
import math
import numpy as np
from PIL import Image
import os
import random
from scipy import misc
from scipy.ndimage import filters


from myloader import CUBDSMULTILBL


UpSample = nn.Upsample(scale_factor=4)
SoftMax = nn.Softmax(dim=1)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def adjust_learning_rate(optimizer, epoch, lr_epoch_change):
    """Sets the learning rate to the initial LR decayed by 10 every lr_epoch_change epochs"""
    lr = args.lr * (0.1 ** (epoch // lr_epoch_change))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def write_results(prec, name, file_name, f):
    f.write(name + '=[')
    for i,p in enumerate(prec):
        if i>0:
            f.write(', ')
        f.write(str(p))
    f.write('];\n')

def write_total_results(file_name_results, prec_tr1, prec_tr2, prec_val1, prec_val2, lr_val, comments):
    f = open(file_name_results, 'w')
    f.write(comments)
    write_results(prec_tr1, 'tr1', file_name_results, f)
    write_results(prec_tr2, 'tr2', file_name_results ,f)
    write_results(prec_val1, 'val1', file_name_results, f)
    write_results(prec_val2, 'val2', file_name_results, f)
    write_results(lr_val, 'lr_val', file_name_results, f)
    f.write('T=[tr1;tr2;val1;val2];\n')
    f.write('h=plot(1:size(T,2),T, \'LineWidth\', 2);\n')
    f.write('grid on;\n')
    f.write('set(h, {\'color\'}, {[.1 .1 .1]; [.9 .1 .1]; [.9 .9 .1]; [.1 .1 .95]});\n')
    f.write('legend(\'tr1\',\'tr2\', \'val1\', \'val2\');\n')
    f.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_time_str():
    return datetime.datetime.now().strftime("%Y_%m_%d_%I_%M%p")

def model_loader(path):
    #model_st = torch.load('alexnet_lrchange_-1_batchsize_128_ImagenetTransform_True_lr_[0.01].pth.tar')
    print('loading ...')
    model_st = torch.load(path)
    model_name = model_st['model_name']
    model = models.__dict__[model_name]()
    classifiers = model_st['classifiers']
    if model_name == 'alexnet' or model_name.startswith('vgg16'):
        model.classifier._modules['6'] = classifiers['label']
    else:
        model.fc = classifiers['label']
    if model_st['data_parallel']:
        if model_name == 'alexnet' or model_name.startswith('vgg16'):
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model)
    model.load_state_dict(model_st['model_st'])
    model = model.cuda()
    time.sleep(1)
    tranform_ImageNet = model_st['tranform_ImageNet']
    return model, classifiers, model_name, tranform_ImageNet

def in_range_prob(prob, low, high):
    if prob.data[0] < high and prob.data[0] > low:
        return True
    else:
        return False


def tensor_normalize_(tensor,m,s):
    for t, m, s in zip(tensor, m, s):
        t.add_(m).div_(s)
    return tensor


def tensor_normalize(tensor,m,s):
    for t, m, s in zip(tensor, m, s):
        t.add(m).div(s)
    return tensor




def get_input(train_loader, id, req_grad=False):
    for i, (input_orig, target) in enumerate(train_loader):
        if i == id:
            vinput1 = Variable(input_orig.cuda(), req_grad)
            vinput_orig = Variable(input_orig.cuda())
            target = target.cuda()
            break
    return vinput1, target



def get_inputs(train_loader, ids, req_grad=False):
    retx = list()
    rety = list()
    ids.sort()
    idx = 0
    for i, (input_orig, target) in enumerate(train_loader):
        if i == ids[idx]:
            retx.append(Variable(input_orig.cuda(), req_grad))
            #vinput_orig = Variable(input_orig.cuda())
            rety.append(target.cuda())
            idx += 1
            if idx >= len(ids):
                break
    return retx, rety


def my_save_image(lst, fl_name ,do_normalize=True, up_scale=1):
    sz = len(lst)
    w = lst[0].size()[2]
    r = Tensor(sz,3,w*up_scale,w*up_scale)
    if up_scale != 1:
        U = nn.Upsample(scale_factor=up_scale)
    else:
        U = lambda  x : x
    for i in range(sz):
        r[i] = U(lst[i])
    save_image(r, fl_name, normalize=do_normalize)
    r = None

def is_cuda():
    cuda = torch.cuda.is_available()
    TensorL = torch.cuda.LongTensor if cuda else torch.LongTensor
    TensorD = torch.cuda.DoubleTensor if cuda else torch.DoubleTensor
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    TensorB = torch.cuda.ByteTensor if cuda else torch.ByteTensor
    return cuda, Tensor, TensorL, TensorD, TensorB

def get_top_two_classes(D, I, y_second=None):
    _, top_two = D(I).topk(2)
    if y_second == None:
        ys = top_two
    else:
        ys = TensorL(1, 2)
        ys[0, 1] = y_second[0]
        ys[0, 0] = top_two[0, 0]
    second = ys[0, 1].item()
    vsecond = Variable(TensorL(1))
    vsecond[0] = second
    ystar = TensorL(1)
    ystar[0] = ys[0, 0]
    vystar = Variable(ystar)
    return vystar, vsecond

def get_image_samples(cnum, dir=None):
    cnum += 1
    if dir == None:
        dir = '/home/sfegh001/amir/pytorch/alexmodel/cropped_train_val/train'
    ss = os.listdir(dir)
    ss.sort()
    aa = ss[cnum]
    fls = os.listdir(os.path.join(dir, aa))
    imgs = list()
    means = [0.4725, 0.4723, 0.4072]
    stds = [0.1126, 0.1052, 0.1403]
    normalize = transforms.Normalize(mean=means, std=stds)
    tr = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),normalize])
    for i in range(4):
        pth = os.path.join(dir,aa,fls[i])
        img = Image.open(pth)
        #data = np.asarray( img, dtype="int32" )
        t = Tensor(tr(img).cuda())
        imgs.append(t)
    return imgs


def revnormalize(tensor):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    #means = [0.4725, 0.4723, 0.4072]
    #stds = [0.1126, 0.1052, 0.1403]
    #means, stds = get_means_stds()
    i = 0;
    ret = Tensor(tensor.size()).fill_(0)
    for t, m, s in zip(tensor, means, stds):
        print('hello, ', i)
        ret[i,:,:] = t.data.mul(s).add(m)
        i+= 1
    return ret

def sorted_indices_to_tuple(ind, max_pixels=32):
    rows = ind.size()[0]
    q = tuple()
    for i in range(min(max_pixels, ind.size()[1])):
        t = tuple()
        for j in range(rows):
            t += (ind[j,i],)
        q += (t,)
        #q += ((ind[:, i].tolist()),)
    return q

def get_sorted_indices(tens, descend=False):
    tot_val = 1
    _, ind = tens.view(1, -1).sort(descending=descend)
    if isinstance(ind, Variable):
        ind = ind.data
    for val in tens.size():
        tot_val *= val
    ret = TensorL(len(tens.size()), tot_val).fill_(0)
    for i, val in enumerate(tens.size()):
        ret[i] = ind / (int(tot_val // val))
        ind -= (ret[i] * int((tot_val // val)))
        tot_val /= val
    return ret

def get_binary_mask(tens, descend, max_pixels=32):
    ind = get_sorted_indices(tens, descend)
    tuples = sorted_indices_to_tuple(ind, max_pixels)
    bin_mask = TensorL(tens.size()).fill_(0)
    for q in tuples:
        bin_mask[q] = 1
    return bin_mask

def create_gaussian_convolver(n, sigma):
    kernel = Tensor(1,1,n,n).fill_(0)
    x = n//2
    y = n //2
    pads = n//2
    for i in range(n):
        for j in range(n):
            kernel[0,0, i,j] = 1 / (2*3.1415926536* sigma * sigma) * math.exp((math.pow((i-x),2) + math.pow((j-y),2)) / (-2*sigma *sigma))
    conv = nn.Sequential(nn.Conv2d(1,1,kernel_size=(n,n),padding=pads,bias=False))
    conv[0].weight.data = kernel
    if cuda:
        conv = conv.cuda()
    return conv

def apply_gaussian(img, n, sigma):
    conv = create_gaussian_convolver(n,sigma, width=img.size()[2])
    #m = img.size()[2] - (n-1)
    outp = Tensor(img.size()).fill_(0)
    for i in range(3):
        b = img[0,i,:,:]
        outp[0,i,:,:] = conv(b.unsqueeze(0).unsqueeze(0))
    return outp

def prep_input_mnist(input_orig, target, D):
    vinput1 = Variable(input_orig.cuda(), requires_grad=True)



def prep_input_cub(input_orig, target, D):
    vinput1 = Variable(input_orig.cuda(), requires_grad=True)
    #g_loss_phase0 = list()
    f_layer = nn.AdaptiveAvgPool2d((224, 224))
    input = f_layer(input_orig)
    input_var = torch.autograd.Variable(input.data.cuda(), requires_grad=True) if cuda else torch.autograd.Variable(
        input, requires_grad=True)
    target_var = torch.autograd.Variable(target.cuda()) if cuda else torch.autograd.Variable(target)
    yhat = D(input_var)
    _, clas = yhat.topk(2)
    prob,_=yhat.topk(200)
    prob = SoftMax(prob)
    return input_var, vinput1, target_var, prob, clas

def ztot(z):
    tensor = ntot(z)
    tensor = tensor.resize(1,tensor.size()[1],1,1)
    t = Tensor(tensor.clone())
    return t

def show_pred_class(D, imgs, cls, printitng=False,plotting=True):
    plt_vals = list()
    lbls = list()
    for i in cls:
        plt_vals.append(list())
        lbls.append(str(i.item()))
    for img in imgs:
        yy = D(img)
        if abs(yy.sum().item() - 1) > 1e-6:
            yy = SoftMax(yy)
        if printitng:
            print(yy[0,cls].item())
        if plotting:
            for i in range(cls.size()[0]):
                plt_vals[i].append(yy[0,cls[i]])
    if plotting:
        if len(plt_vals) == 1:
            plot_loss([plt_vals],lbls)
        else:
            plot_loss(plt_vals, lbls)


def show_pred_imgs(D,imgs, tk=2):
    for img in imgs:
        yy = D(img)
        if abs(yy.sum().item() - 1) > 1e-6:
            yy = SoftMax(yy)
        print(yy.topk(tk))

def show_pred_z(D,G,zs, tk=2):
    for z in zs:
        if str(type(z)).find('torch')>=0:
            vz = Variable(z.data)
        else:
            vz = Variable(ztot(z))
        yhat = D(G(vz))
        #print(yhat.sum())
        if abs(yhat.sum().item() - 1) > 1e-8:
            yhat = SoftMax(yhat)
        prob, clas = yhat.topk(tk)
        print('probs: ' ,prob, ' , classes: ', clas)


def ntot(z_np):
    tensor = Tensor(z_np)
    return tensor

def z_images(z_list, G, file_name='z_imgs.jpg', scale_fact=4):
    UpSample = nn.Upsample(scale_factor=scale_fact)
    imgs = list()
    imgs_orig = list()
    for z in z_list:
        var = Variable(Tensor(z))
        var = var.resize(1,z_list[0].shape[1],1,1)
        imgs.append(UpSample(G(var)))
        imgs_orig.append(G(var))
    my_save_image(imgs, file_name)
    return imgs_orig


def z_images_vae(z_list, G, file_name='z_imgs.jpg', scale_fact=4):
    UpSample = nn.Upsample(scale_factor=scale_fact)
    imgs = list()
    imgs_orig = list()
    for z in z_list:
        var = Variable(Tensor(z))
        var = var.resize(1,z_list[0].shape[1],1,1)
        imgs.append(UpSample(G(var)))
        imgs_orig.append(G(var))
    my_save_image(imgs, file_name)
    return imgs_orig


def learn_new_z_signs(G, img_orig, z_np_arr, my_lr_lst, loss_fns, weights, iters=10, keep_label=False, true_label=None, Dis=None, normalize=None):
    img = Variable(img_orig.data.clone().cuda(), requires_grad=False)
    D = Dis
    z_temp = Tensor(z_np_arr)
    z = z_temp.clone().resize(1,z_temp.size()[1],1,1)
    vz_orig = Variable(z.clone().cuda(), requires_grad = True)
    #vz_orig.resize(1,100,1,1)
    all_zs = list()
    g_all_loss = list()
    #UpSample4 = nn.functional.interpolate(scale_factor=4)
    f_layer = nn.AdaptiveAvgPool2d((224, 224))
    for my_lr in my_lr_lst:
        z = Variable(vz_orig.data.clone().cuda(), requires_grad=True)
        #z = z.resize(1,100,1,1)
        g_loss = list()
        optimizer_z = optim.Adam([z], lr=my_lr)
        for ii in range(iters):
            G.zero_grad()
            optimizer_z.zero_grad()
            #print('zs: ', z.size())
            Gz = G(z)
            Gz = f_layer(nn.functional.interpolate(G(z),scale_factor=4))
            loss = loss_fns[0](img, Gz) * weights[0]
            for j in range(1,len(loss_fns)):
                loss += loss_fns[j](img, Gz) * weights[j]
            loss.backward()
            g_loss.append(loss.item())
            optimizer_z.step()
            if ii == 0:
                prev = z.data.abs().sum()
            else:
                cur = z.data.abs().sum()
                #print('cur diff : ', cur.item() - prev.item())
                prev = cur
            if keep_label and ii > iters//2:
                #gz = G(z)
                #gz.requires_grad=True
                #z = Variable(z.data.clone().cuda(), requires_grad=True)
                optimizer_keep_label = optim.Adam([z], lr=my_lr)
                ygz = F.log_softmax(D(Gz), dim=1)
                p,c = ygz.topk(1)
                #print('keep, cur_labl: ', c)
                label_iter = 0
                while c[0,0].item() != true_label.item() and label_iter < 100:
                    #print('k: ', label_iter)
                    label_iter += 1
                    crit = nn.NLLLoss()
                    vY = Variable(true_label.cuda(), requires_grad=False)
                    gz = f_layer(nn.functional.interpolate(G(z), scale_factor=4))
                    ygz = F.log_softmax(D(gz), dim=1)
                    loss2 = crit(ygz,vY)
                    optimizer_keep_label.zero_grad()
                    G.zero_grad()
                    D.zero_grad()
                    loss2.backward(retain_graph=True)
                    optimizer_keep_label.step()
                    #gz.requires_grad = True
                    #ygz = F.log_softmax(D(gz))
                    p, c = ygz.topk(1)
                    #print(c[0,0].item())
                #z = Variable(z2.data.clone().cuda(), requires_grad=True)
        g_all_loss.append(g_loss)
        all_zs.append(z)
    return all_zs, g_all_loss


def learn_new_z_fmnist(G, img_orig, z_np_arr, my_lr_lst, loss_fns, weights, iters=10, keep_label=False, true_label=None, Dis=None, normalize=None):
    img = Variable(img_orig.data.clone().cuda(), requires_grad=False)
    D = Dis
    z_temp = Tensor(z_np_arr)
    z = z_temp.clone().resize(1,z_temp.size()[1],1,1)
    vz_orig = Variable(z.clone().cuda(), requires_grad = True)
    #vz_orig.resize(1,100,1,1)
    all_zs = list()
    g_all_loss = list()
    #UpSample4 = nn.functional.interpolate(scale_factor=4)
    #f_layer = nn.AdaptiveAvgPool2d((224, 224))
    for my_lr in my_lr_lst:
        z = Variable(vz_orig.data.clone().cuda(), requires_grad=True)
        #z = z.resize(1,100,1,1)
        g_loss = list()
        optimizer_z = optim.Adam([z], lr=my_lr)
        for ii in range(iters):
            G.zero_grad()
            optimizer_z.zero_grad()
            #print('zs: ', z.size())
            Gz = G(z)
            #Gz = f_layer(nn.functional.interpolate(G(z),scale_factor=4))
            loss = loss_fns[0](img, Gz) * weights[0]
            for j in range(1,len(loss_fns)):
                loss += loss_fns[j](img, Gz) * weights[j]
            loss.backward()
            g_loss.append(loss.item())
            optimizer_z.step()
            if ii == 0:
                prev = z.data.abs().sum()
            else:
                cur = z.data.abs().sum()
                #print('cur diff : ', cur.item() - prev.item())
                prev = cur
            if keep_label and ii > iters//2:
                #gz = G(z)
                #gz.requires_grad=True
                #z = Variable(z.data.clone().cuda(), requires_grad=True)
                optimizer_keep_label = optim.Adam([z], lr=my_lr)
                ygz = F.log_softmax(D(Gz), dim=1)
                p,c = ygz.topk(1)
                #print('keep, cur_labl: ', c)
                label_iter = 0
                while c[0,0].item() != true_label.item() and label_iter < 100:
                    #print('k: ', label_iter)
                    label_iter += 1
                    crit = nn.NLLLoss()
                    vY = Variable(true_label.cuda(), requires_grad=False)
                    gz = G(z) #f_layer(nn.functional.interpolate(G(z), scale_factor=4))
                    ygz = F.log_softmax(D(gz), dim=1)
                    loss2 = crit(ygz,vY)
                    optimizer_keep_label.zero_grad()
                    G.zero_grad()
                    D.zero_grad()
                    loss2.backward(retain_graph=True)
                    optimizer_keep_label.step()
                    #gz.requires_grad = True
                    #ygz = F.log_softmax(D(gz))
                    p, c = ygz.topk(1)
                    #print(c[0,0].item())
                #z = Variable(z2.data.clone().cuda(), requires_grad=True)
        g_all_loss.append(g_loss)
        all_zs.append(z)
    return all_zs, g_all_loss


def learn_new_z_mnist(G, img_orig, z_np_arr, my_lr_lst, loss_fns, weights, iters=10, keep_label=False, true_label=None, Dis=None, normalize=None):
    img = Variable(img_orig.data.clone().cuda(), requires_grad=False)
    D = Dis
    z_temp = Tensor(z_np_arr)
    z = z_temp.clone().resize(1,z_temp.size()[1],1,1)
    vz_orig = Variable(z.clone().cuda(), requires_grad = True)
    #vz_orig.resize(1,100,1,1)
    all_zs = list()
    g_all_loss = list()
    for my_lr in my_lr_lst:
        z = Variable(vz_orig.data.clone().cuda(), requires_grad=True)
        #z = z.resize(1,100,1,1)
        g_loss = list()
        optimizer_z = optim.Adam([z], lr=my_lr)
        for ii in range(iters):
            G.zero_grad()
            optimizer_z.zero_grad()
            #print('zs: ', z.size())
            Gz = G(z)
            loss = loss_fns[0](img, Gz) * weights[0]
            for j in range(1,len(loss_fns)):
                loss += loss_fns[j](img, Gz) * weights[j]
            loss.backward()
            g_loss.append(loss.item())
            optimizer_z.step()
            if ii == 0:
                prev = z.data.abs().sum()
            else:
                cur = z.data.abs().sum()
                #print('cur diff : ', cur.item() - prev.item())
                prev = cur
            if keep_label and ii > iters//2:
                #gz = G(z)
                #gz.requires_grad=True
                #z = Variable(z.data.clone().cuda(), requires_grad=True)
                #optimizer_keep_label = optim.Adam([z], lr=my_lr)
                ygz = D(G(z))
                p,c = ygz.topk(1)
                #print('keep, cur_labl: ', c)
                label_iter = 0
                while c[0,0].item() != true_label.item() and label_iter < 100:
                    #print('k')
                    label_iter += 1
                    #crit = nn.CrossEntropyLoss()
                    crit = nn.NLLLoss()
                    vY = Variable(true_label.cuda(), requires_grad=False)
                    loss2 = crit(ygz,vY)
                    #optimizer_keep_label.zero_grad()
                    optimizer_z.zero_grad()
                    G.zero_grad()
                    D.zero_grad()
                    loss2.backward()
                    #optimizer_keep_label.step()
                    optimizer_z.step()
                    gz = G(z)
                    #gz.requires_grad = True
                    ygz = D(gz)
                    p, c = ygz.topk(1)
                    #print(c[0,0].item())
                #z = Variable(z2.data.clone().cuda(), requires_grad=True)
        g_all_loss.append(g_loss)
        all_zs.append(z)
    return all_zs, g_all_loss


def get_overlayed_image(x, c, gray_factor_bg=0.3):
    '''
    For an image x and a relevance vector c, overlay the image with the
    relevance vector to visualise the influence of the image pixels.
    '''
    imDim = x.shape[0]
    if np.ndim(c) == 1:
        c = c.reshape((imDim, imDim))
    if np.ndim(x) == 2:  # this happens with the MNIST Data
        x = 1- np.dstack((x, x, x)) * gray_factor_bg  # make it a bit grayish
    elif np.ndim(x) == 3:  # this is what happens with cifar data
        x = color.rgb2gray(x)
        x = 1 - (1 - x) * 0.5
        x = np.dstack((x, x, x))
    alpha = 0.8
    # Construct a colour image to superimpose
    im = plt.imshow(c, cmap=cm.bwr, vmin=-np.max(np.abs(c)), vmax=np.max(np.abs(c)), interpolation='nearest')
    color_mask = im.to_rgba(c)[:, :, [0, 1, 2]]
    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = color.rgb2hsv(x)
    color_mask_hsv = color.rgb2hsv(color_mask)
    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def get_overlayed_image_gif(x, c, gray_factor_bg=0.3, fl_name='graphics', N=9):
    '''
    For an image x and a relevance vector c, overlay the image with the
    relevance vector to visualise the influence of the image pixels.
    '''
    imDim = x.shape[0]
    if np.ndim(c) == 1:
        c = c.reshape((imDim, imDim))
    if np.ndim(x) == 2:  # this happens with the MNIST Data
        x = 1- np.dstack((x, x, x)) * gray_factor_bg  # make it a bit grayish
    elif np.ndim(x) == 3:  # this is what happens with cifar data
        x = color.rgb2gray(x)
        x = 1 - (1 - x) * 0.5
        x = np.dstack((x, x, x))
    alpha = 0.8
    # Construct a colour image to superimpose
    for k in range(1,N+1):
        plt.close()
        im = plt.imshow(c, cmap=cm.bwr, vmin=-k*1.0/N*np.max(np.abs(c)), vmax=k*1.0 /N*np.max(np.abs(c)), interpolation='nearest')
        color_mask = im.to_rgba(c)[:, :, [0, 1, 2]]
        # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
        img_hsv = color.rgb2hsv(x)
        color_mask_hsv = color.rgb2hsv(color_mask)
        # Replace the hue and saturation of the original image
        # with that of the color mask
        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
        img_masked = color.hsv2rgb(img_hsv)
        plt.imshow(img_masked)
        plt.savefig(fl_name + '_' + str(k) + '.jpg')
    return img_masked


def get_overlayed_image_modified(x, c, gray_factor_bg=0.3):
    '''
    For an image x and a relevance vector c, overlay the image with the
    relevance vector to visualise the influence of the image pixels.
    '''
    imDim = x.shape[0]
    if np.ndim(c) == 1:
        c = c.reshape((imDim, imDim))
    if np.ndim(x) == 2:  # this happens with the MNIST Data
        x = 1- np.dstack((x, x, x)) * gray_factor_bg  # make it a bit grayish
    elif np.ndim(x) == 3:  # this is what happens with cifar data
        x = x.clip(max=1, min=-1)
        x = color.rgb2gray(x)
        x = 1 - (1 - x) * 0.5
        x = np.dstack((x, x, x))
    alpha = 0.3
    c -= c.min()
    # Construct a colour image to superimpose
    im = plt.imshow(c, cmap=cm.bwr, vmin=-np.max(np.abs(c)), vmax=np.max(np.abs(c)), interpolation='nearest')
    color_mask = im.to_rgba(c)[:, :, [0, 1, 2]]
    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = color.rgb2hsv(x)
    color_mask_hsv = color.rgb2hsv(color_mask)
    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def make_in_range(t, a, b):
    K = (t.clone()-t.min()) / (t.max()-t.min())
    ret = t.clone().fill_(0)
    ret = a + K * (b-a)
    return ret


#def tval_to_float(tensor_list)
def tensor_normalize(t):
    t1 = t.clone()
    t1 -= t1.min()
    if abs(t1.max().item()) <= -1e-12:
        raise NameError('Tensor Normalize, division by zero')
    t1 /= t1.max()
    return t1


def tensor_normalize_(t1):
    #t1 = t.clone()
    t1 -= t1.min()
    t1 /= t1.max()
    return t1


def tensor_list_0_1_normalize(t_list):
    ret = list()
    for t in t_list:
        t1 = tensor_normalize(t)
        ret.append(t1)
    return ret


def z_detection_class_probs(zs, D,G, topk=2, prn=True ):
    if str(type(zs[0])).find('torch') < 0:
        new_zs = list()
        for z in zs:
            new_zs.append(ztot(z))
        zs = new_zs
    pr = list()
    cl = list()
    for z in zs:
        yhat = D(G(z))
        _,cla = yhat.topk(topk)
        cl.append(cla)
        pra,_ = yhat.topk(topk)
        pr.append(pra)
    ret = list()
    if prn:
        print(pr,cl)
    for x in zip(cl, pr):
        ret.append(x)
    return ret


def tv_norm(fake, x, beta=2.):
    assert(x.size(0) == 1)
    img = x[0]
    dy = img - img # set size of derivative and set border = 0
    dx = img - img
    dy[:,1:,:] = -img[:,:-1,:] + img[:,1:,:]
    dx[:,:,1:] = -img[:,:,:-1] + img[:,:,1:]
    return ((dx.pow(2) + dy.pow(2)).pow(beta/2.)).sum()


def learn_new_z_cub(G, img, z_np_arr, my_lr_lst, iters, img_id, loss_fn, normalize=None, D=None, target_var=None, loss_weights=None, correct=None):
    if normalize != None:
        print('normalize')
    if correct == None:
        correct = ''
    weights = list()
    if loss_weights==None:
        for i in range(len(loss_fn)):
            weights.append(1.0)
    else:
        weights= loss_weights
    loss_all = list()
    final_z = None
    all_z = list()
    img_save_interval = iters // 10
    if D != None:
        f_layer = nn.AdaptiveAvgPool2d((224, 224))
        crit = nn.CrossEntropyLoss()
        #yhat = D(f_layer(img))
    #_,ytrue = yhat.topk(1)
    temp = Variable(Tensor(z_np_arr))
    z = temp.clone().resize(1, 120, 1, 1)
    vz_orig = Variable(z.clone().cuda(), requires_grad=True)
    z_orig = Variable(vz_orig.data.clone().cuda(), requires_grad=True)
    #z_orig.resize_(1,z_orig.size()[1],1,1)
    prn_interval = iters //5
    for my_lr in my_lr_lst:
        print('lr: ', my_lr)
        g_loss_phase0 = list()
        z = Variable(z_orig.data.clone(), requires_grad=True)
        optimzer_z = optim.Adam([z], lr=my_lr)
        lst_imgs = list()
        lst_imgs.append(img.data)
        for iter_z in range(iters):
            if iter_z%prn_interval == 1:
                print('lr: ', my_lr, ', iter: ', iter_z)
            G.zero_grad()
            optimzer_z.zero_grad()
            Gz = G(z)
            #Gz = tensor_normalize(Gz, normalize.mean, normalize.std)
            Gz_up = UpSample(Gz)
            if iter_z % img_save_interval == 0:
                lst_imgs.append(Gz_up.data.clone())
            loss = None
            for ii, cur_loss_fn in enumerate(loss_fn):
                loss = weights[ii] * cur_loss_fn(img, Gz_up)#(img - Gz_up).pow(2).mean()
                loss.backward(retain_graph=True)
                #print('z_' ,str(ii), str(z.grad.data.abs().sum()), ' -> loss: ', loss.data)
            #print('z before backwardcd(), ', z.requires_grad)

            # print('row: {0}, col: {1}, loss: {2} '.format(row,col,loss.data[0]))
            #print('loss: {0} '.format(loss.data[0]))
            if D!= None:
                yhat = D(f_layer(Gz_up))
                loss = crit(yhat, target_var)
                loss.backward(retain_graph=True)
            g_loss_phase0.append(loss.data[0].item())
            optimzer_z.step()
        my_save_image(lst_imgs, 'results7/' + str(iters) + '_' + get_function_name(loss_fn) + '_' + str(z_orig.data.mean().item()) + '_' +
                      str(img_id) + '_{:0.9f}'.format(my_lr) + '_' + str(correct) + '_losslen_' + str(len(loss_fn)) + '.jpg')
        loss_all.append(g_loss_phase0)
        all_z.append(z)
        if g_loss_phase0[-1] <= loss_all[-1][-1]:
            final_z = Variable(z.data.clone(), requires_grad = True)
        #if prob.data[0] > 0.9 or prob.data[0] < 0.5:
        #    break
        #break
    print('dot product zorig, finalz: ', math.degrees(math.acos(
        (z_orig[0, :, 0, 0].dot(final_z[0, :, 0, 0]) / (
        z_orig[0, :, 0, 0].norm(2) * final_z[0, :, 0, 0].norm(2))).item())))
    return final_z, loss_all, all_z


def get_GAN_explanation_mnist(G, D, zs, init_z, I_org, mycrit, final_crit, z_distance_name,distance_name, iter, y_seconds=None):
    zstar = Variable(zs.data.clone(), requires_grad=True)
    Gzinit = Variable(Tensor(init_z), requires_grad=True)
    Gzinit = Gzinit.resize(1, zstar.size()[1], 1, 1)
    Gzinit = G(Gzinit)
    #Gzinit = UpSample(Gzinit)
    Up = nn.Upsample(scale_factor=2)
    Gz0 = G(zstar)
    optimizer = optim.Adam([zstar], lr=0.01)
    for ii in range(10):
        G.zero_grad()
        optimizer.zero_grad()
        Gz = G(zstar)
        vystar, vsecond = get_top_two_classes(D, Gz, y_seconds)
        loss = mycrit(D, Gz, vystar, vsecond, final_crit, more=-1)
        G.zero_grad()
        D.zero_grad()
        loss.backward()
        dz = zstar.grad.data  # / zstar.grad.data.norm(2)
        # zstar = Variable(zstar.data - 0.1 * dz.data, requires_grad=True)
        optimizer.step()
        # print('1,2:', zstar.grad.data.max(),', ', zstar.grad.data.min(), ', ',zstar.grad.data.sum(), ', ' , zstar.grad.data.norm(2) )
    samples = get_image_samples(vsecond.data[0].item())
    lr_list = [0.1, 0.01, 1]
    for lr in lr_list:
        new = G(zstar).data
        diff = new - Gz0.data
        fl_name = 'results7/' + str(vystar.item()) + '_' + str(vsecond.data[0].item()) + '_' + str(init_z.sum()) + '_' + \
                  str(zs.norm(2).item()) + '_' + str(lr) + '_' + z_distance_name + '_' + distance_name + '_' + str(
            iter) + \
                  'more_1_' + '.jpg'
        # print(revnormalize(Up(I_org.data).squeeze(0)).abs().sum() - I_org.data.abs().sum())
        save_lst = [Up(Gzinit), Up(Gz0), Up(new), Up(diff), Up(I_org.data)]
        for p in save_lst:
            print(p.size())
        #for img_s in samples:
        #    save_lst.append(Up(img_s.unsqueeze(0)))
        my_save_image(save_lst, fl_name)


def get_GAN_explanation(G, D, zs, init_z, I_org, mycrit, final_crit, UpSample, z_distance_name,distance_name, iter, y_seconds=None,final_upsample=None, lr_list = [1,0.1, 0.05, 0.01], Gz_full=None):
    #I_org = input_var
    f_layer = nn.AdaptiveAvgPool2d((224, 224))
    if final_upsample == None:
        Up = lambda x : x
    else:
        Up = final_upsample
    I = Variable(I_org.data)
    #z_orig = Variable(Tensor(z_np_arr), requires_grad=True)
    zstar = Variable(zs.data.clone(), requires_grad=True)
    Gzinit = Variable(Tensor(init_z), requires_grad=True)
    Gzinit = Gzinit.resize(1, zstar.size()[1], 1, 1)
    Gzinit = G(Gzinit)
    Gzinit = UpSample(Gzinit)
    Gzinit = f_layer(Gzinit)
    if Gz_full == None:
        Gz0 = G(zstar)
        Gz0 = UpSample(Gz0)
        Gz0 = f_layer(Gz0)
        optimizer = optim.Adam([zstar], lr=0.01)
    else:
        Gz0 = f_layer(UpSample(Gz_full))
        optimizer = optim.Adam([Gz_full], lr=0.01)
    for ii in range(10):
        G.zero_grad()
        optimizer.zero_grad()
        Gz = G(zstar)
        Gz = UpSample(Gz)
        Gz = f_layer(Gz)
        vystar, vsecond = get_top_two_classes(D, Gz, y_seconds)
        loss = mycrit(D, Gz, vystar, vsecond, final_crit, more=-1)
        G.zero_grad()
        D.zero_grad()
        loss.backward()
        dz = zstar.grad.data #/ zstar.grad.data.norm(2)
        #zstar = Variable(zstar.data - 0.1 * dz.data, requires_grad=True)
        optimizer.step()
        #print('1,2:', zstar.grad.data.max(),', ', zstar.grad.data.min(), ', ',zstar.grad.data.sum(), ', ' , zstar.grad.data.norm(2) )
    samples = get_image_samples(vsecond.data[0].item())
    for lr in lr_list:
        new = f_layer(UpSample(G(zstar +  lr *dz).data))
        diff = new   - Gz0.data
        fl_name = 'results7/' + str(vystar.item()) + '_' + str(vsecond.data[0].item()) + '_' +  str(init_z.sum()) + '_' + \
                  str(zs.norm(2).item()) + '_' + str(lr) + '_' + z_distance_name + '_' + distance_name + '_' + str(iter) +\
                  'more_1_' + '.jpg'
        #print(revnormalize(Up(I_org.data).squeeze(0)).abs().sum() - I_org.data.abs().sum())
        save_lst = [ Up(Gzinit) , Up(Gz0), Up(new), Up(diff), Up(I_org.data)]
        for img_s in samples:
            save_lst.append(Up(img_s.unsqueeze(0)))
        my_save_image(save_lst, fl_name)

'''
def get_GAN_explanation(G, D, zs, I_org, loss_fn, lr_list = [0.1], iters_list = [1]):
    for ii in iters_list:
        I = Variable(I_org.data)
        zstar = Variable(zs.data, requires_grad=True)
        for iter in range(ii):
            G.zero_grad()
            Gz = G(zstar)
            yhat = D(Gz)
            _, ystar = D(I_org).topk(1)
            loss = loss_fn(yhat, ystar)
            D.zero_grad()
            loss.backward()
            dz = z.grad.data
            for lr in lr_list:
                fl_name = 'results4/' + str(ystar.item()) + '_' + str(iter) + '_' + str(lr) + '.jpg'
                save_image(G(zstar + dz).data - Gz.data, fl_name)

'''

def my_contra_crit_loss_func(D, Igz, ytrue, ysecond, crit,  more=-1, yh=None):
    yhat = D(Igz)
    return more * crit(yhat, ytrue) - more * crit(yhat, ysecond)


def plot_pr_cl(pr_cl):
    #x=[1,2,3]
    #y=[9,8,7]
    x = list()
    y = list()
    lbl = list()
    for i in range(len(pr_cl)):
        x.append(i)
        y.append(pr_cl[i][0])
        lbl.append(pr_cl[i][1])
    pyplot.plot(x,y)
    for a,b,c in zip(x, y, lbl):
        pyplot.text(a, b, c)
    pyplot.show()


def extract_tensor_item(t1_):
    if str(type(t1_)).find('torch') >=0:
        t1 = t1_.item()
    else:
        t1 = t1_
    return t1


def plot_loss(gs, legend=None):
    x = list()
    for i in range(len(gs[0])):
        x.append(i)
    plt.close()
    print(type(legend))
    leg = list()
    markers = [".", "v", "*", "o", "8", "s", "+", "x", "D", "h"]
    for i, ll in enumerate(gs):
        if legend!=None:
            lbl = str(legend[i])
        else:
            lbl = ''
        hdl, = plt.plot(x,ll, label=lbl, linewidth=1.6, marker=markers[i], markevery=len(ll)//10)
        leg.append(hdl)
    plt.legend(handles=leg)
    plt.show()


def get_pr_cl_class_change(pr_cl, t1_, t2_):
    t1 = extract_tensor_item(t1_)
    t2 = extract_tensor_item(t2_)
    last = False
    idx = 0
    for idx,x in enumerate(pr_cl):
        if x[1] == t1:
            continue
        else:
            break
    for j in range(idx, len(pr_cl)):
        if pr_cl[j][1] == t2:
            return j
    raise 'No target class'


def make_more_lbl_img(D,inp_img, opt_lr, tot_iters, loss, target, sgn=1.0):
    imgs = list()
    pr_cl = list()
    #dimg_cuml_abs = list()
    dimg_cuml = list()
    dimg_cuml.append(Variable(Tensor(inp_img.size()).fill_(0)))
    #dimg_cuml_abs.append(Variable(Tensor(inp_img.size()).fill_(0)))
    img = Variable(inp_img.data.clone().cuda(), requires_grad=True)
    optimizer = optim.Adam([img], lr=opt_lr)
    for p in D.parameters():
        p.requires_grad=False
    for iter in range(tot_iters):
        yhat = D(img)
        pr,cl = yhat.topk(1)
        if abs(yhat.sum().item() - 1) > 1e-5:
            #print('doing softmax')
            pr = SoftMax(yhat)
            pr,_ = pr.topk(1)
        pr_cl.append((pr.item(),cl.item()))
        imgs.append(Variable(img.data.clone().cuda(), requires_grad=True))
        loss_val = sgn * loss(yhat, target.cuda())
        D.zero_grad()
        prev_img = img.clone()
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        dimg_cuml.append(dimg_cuml[-1].clone() + img.clone()-prev_img.clone())
        #dimg_cuml_abs.append(dimg_cuml[-1].clone() + img.data.clone().abs())
    return imgs, pr_cl, dimg_cuml#, dimg_cuml_abs


def get_grad_img_contrastive(model, img, ytrue, crit, ysecond, Softmax=None, more=1, iters=1):
    derives = Variable(Tensor(img.size()).fill_(0))
    bs, nc, w, h = img.size()
    I = img.clone()
    for i in range(iters):
        yhat = model(img)
        _, ys = yhat.topk(2)
        #if ysecond.size() != None:
        if 'Variable' in str(type(ysecond)):
            ys[0,1] = ysecond[0]
        cur_grads = Tensor(2, bs, nc, w, h)
        for j in range(2):
            model.zero_grad()
            X = Variable(img.data.clone(), requires_grad=True)
            yhat = model(X)
            #print(yhat.type(), yhat.size())
            if Softmax != None:
                yhat = Softmax(yhat)
            # print('ysum: ' , yhat.sum())
            loss = crit(yhat, ys[:, j])
            model.zero_grad()
            loss.backward(retain_graph=True)
            cur_grad = X.grad / X.grad.data.norm(2)
            cur_grads[j, :, :, :, :] = cur_grad.data
        over_all = (-more * cur_grads[0, :, :, :, :] + more * cur_grads[1, :, :, :, :])
        over_all = over_all.squeeze(0)
        derives.data += over_all / (i + 1)
        img = Variable(img.data.clone() + over_all )
    diff_img = I - img
    return derives, img, diff_img


#more {-1,1}
#returns the geometricvally average gradient and the final image
def get_grad_img(model, img, ytrue, crit, SoftMax=None, more=1, iters=1):
    ret = Variable(Tensor(img.size()).fill_(0))
    for i in range(iters):
        model.zero_grad()
        X = Variable(img.data.clone().cuda(), requires_grad=True)
        yhat = model(X)
        if SoftMax != None:
            yhat = SoftMax(yhat)
        #print('ysum: ' , yhat.sum())
        loss = crit(yhat, ytrue)
        '''
        if more==1:
            loss = crit(yhat, ytrue)
        else:
            loss = -crit(yhat, ytrue)
        '''
        model.zero_grad()
        loss.backward()
        cur_grad = X.grad / X.grad.data.norm(2)
        ret += cur_grad / (i+1)
        img = Variable(img.data.clone() - more * cur_grad.data)
    return ret, img


#size img: 1x3xwxh
def get_regions(convolver, grad_img, I, max_pixels, fade=True):
    img = Tensor(I.size()).fill_(1)
    width = convolver[0].kernel_size[0]
    height = convolver[0].kernel_size[1]
    for i in range(3):
        cur_msk = Tensor(grad_img.size()[2], grad_img.size()[3]).fill_(0)
        msk_bin = cur_msk.clone()
        #print('grad.size: ', grad_img.size())
        grad_conv = convolver(grad_img[:, i, :, :].unsqueeze(0).abs())
        #print('con: ', grad_conv.size())
        pwr = 1.0
        sorted_indices = get_sorted_indices(grad_conv.data, True)
        #print('ss: ', sorted_indices.size())
        inds = sorted_indices_to_tuple(sorted_indices, max_pixels)
        for q in inds:
            cur_msk[q[2]:q[2] + width, q[3]:q[3] + height] += pwr
            msk_bin[q[2]:q[2] + width, q[3]:q[3] + height] = 1
            if fade:
                pwr -= 0.005
            if msk_bin.sum() > max_pixels:
                break

        img[0, i, :, :] *= (cur_msk.clamp(max=1))
        #print('pwr: ', pwr)
    applied = tensor_to_img_range(I.data.clone()) + tensor_to_img_range(img.clone()) / 2
    return img, applied


def tensor_to_img_range(t):
    range = t.max() - t.min()
    t = t - t.min()
    t /= (range + 1e-5)
    return t


def get_convolver(nc, nco, w, h, Bias=False):
    convolver = nn.Sequential(nn.Conv2d(nc,nco,kernel_size=(w, h), bias=Bias))
    if cuda:
        convolver = convolver.cuda()
    return convolver


def L2_NORM(a,b):
    return (a-b).norm(2)


def L1_NORM(a,b):
    return (a-b).norm(1)


def MSE(a,b):
    return  (a - b).pow(2).mean()
    #return lambda a,b:(a-b).pow(2).mean()


def PSNR(I,K):
    mse = MSE(I,K)
    return  20 * ((I.max()/255)*255).log10() - 10 * (mse + 1e-12).log10()


def get_function_name(func):
    s = str(func)
    ss = s.split(' ')
    if len(ss) == 1:
        return ss[0]
    q = str(ss[1])
    q = str(q)
    if q.find(".<l") > 0:
        q = q[0:q.find(".<l")]
    return q


def pad(x,dim, r=0, c=0):
    n,m = x.size()[0], x.size()[1]
    new_x = torch.zeros([n+2*dim + r, m + 2*dim + c], dtype=x.dtype).fill_(0)
    if cuda:
        new_x = new_x.cuda()
    new_x[dim:dim+n, dim:dim+m] = x.clone()
    return new_x


def trail_pad(x,dim1,dim2):
    n, m = x.size()[0], x.size()[1]
    new_x = torch.zeros([n + 1 * dim1 , m + 1 * dim2], dtype=x.dtype).fill_(0)
    if cuda:
        new_x = new_x.cuda()
    new_x[0:n, 0:m] = x.clone()
    return new_x


#assumes wxhx2
def make_conj(A):
    ret = A.clone()
    ret[:,:,1] *= -1
    return ret


def conj_divide2(x,y):
    denum = y[:,:,0].pow(2) + y[:,:,1].pow(2)
    #denum[denum.abs()< 1e-12] = 1e18
    real = x[:,:,0]*y[:,:,0] + x[:,:,1]* y[:,:,1]
    imag = x[:,:,1]*y[:,:,0] - x[:,:,0] * y[:,:,1]
    ret = Tensor(y.size()).type(y.dtype).fill_(0)
    ret[:,:,0] = real / denum
    ret[:,:,1] = imag / denum
    return ret


# wxhx2
def conj_mult2(x,y):
    #a = x[:,:,0].mm(y[:,:,0])
    #b = x[:,:,1].mm(y[:,:,1])
    a = x[:, :, 0]*  y[:, :, 0]
    b = x[:,:,1] *y[:,:,1]
    ret = torch.zeros([a.size()[0], a.size()[1], 2], dtype=b.dtype)
    if cuda:
        ret = ret.cuda()
    ret[:,:,0] = a- b
    #c = x[:,:,1].mm(y[:,:,0])
    #d = x[:, :, 0].mm(y[:, :, 1])
    c = x[:, :, 1] *y[:, :, 0]
    d = x[:, :, 0] *y[:, :, 1]
    ret[:,:,1] = c + d
    return ret


def my_unroll(kernel, x, padding=True):
    n = x.size()[0]
    m = kernel.size()[0]
    if padding:
        n2 = n + m - 1
        x_new = Tensor(n2, n2).fill_(0)
        pad = (n2 - n) // 2
        x_new[pad:pad + n, pad:pad + n] = x
        x_new[0:pad, pad:pad + n] = x[0:pad, :]
        x_new[pad + n:n2, pad:pad + n] = x[n - pad:n, :]
        x_new[pad:pad + n, 0:pad] = x[:, 0:pad]
        x_new[pad:pad + n, pad + n:n2] = x[:, n - pad:n]
        x = x_new
        n = x.size()[0]
    else:
        x_new = x
    ret = Tensor((n-m+1)**2,n**2).fill_(0)
    for ii in range((n-m+1)):
      i = ii * (n-m + 1)
      for j in range(n-m+1):
           st = ii * n + j
           for kline in range(m):
                #if kline > 0:
                #     st += n-m
                #print('st: {0}, i:{1}, j:{2}, k:{3}'.format(st, i, j, kline))
                ret[i+j,st:st+m] = kernel[kline,:]
                st += n
    if padding:
        w = ret.size()[1] - ret.size()[0]
        w = w // 2
        #print(w)
        #print(w + ret.size()[0])
        return ret[:, w: w + ret.size()[0]]
    return ret


def get_kernel(path=None):
    if path == None:
        st = torch.load('fergus_kernel.pth')
    else:
        st = torch.load(path)
    return st['kernel'].cuda() if cuda else  st['kernel']


def get_padded_rfft(Gx, mn=(1,)):
    if mn == None:
        return torch.rfft(Gx,2,False,False)
    n1,m1 = mn[0] - Gx.size()[0], mn[1] - Gx.size()[1]
    Gxpad = trail_pad(Gx,n1,m1)
    return torch.rfft(Gxpad, 2, False, False)


def get_deconv_freq(y, kernel, we = 0.002):
    #kernel = get_kernel()
    y = y.type(kernel.dtype)
    GX = get_padded_rfft(Tensor([[-1,1]]).type(torch.cuda.DoubleTensor), tuple(y.size()[2:]))
    GY = get_padded_rfft(Tensor([[1], [-1]]).type(torch.cuda.DoubleTensor), tuple(y.size()[2:]))
    ret = y.clone().fill_(0)
    F = get_padded_rfft(kernel.type(torch.cuda.DoubleTensor), tuple(y.size()[2:]))
    Fc = make_conj(F)
    GXc = make_conj(GX)
    GYc = make_conj(GY)
    #print('fc, F, GX, GY', Fc.size(), F.size(), GXc.size(), GY.size())
    A = conj_mult2(Fc, F) + we * (conj_mult2(GXc, GX) + conj_mult2(GYc, GY))
    #return A
    #y2 = torch.zeros([kernel.size()[0], kernel.size()[1]], dtype=kernel.dtype)
    y2 = y.type(kernel.dtype).clone()
    if cuda:
        y2 = y2.cuda()
    #print(y2.sum())
    y2 = y2.squeeze(0)
    for i in range(3):
        Y = torch.rfft(y2[i,:,:], 2, False, False)
        b = conj_mult2(Fc , Y)
        X = conj_divide2(b, A)
        x = torch.irfft(X,2,False,False)
        ret[0,i,:,:] = x
    return ret


def FFT_distance(distance,deconv_func, kernel_path=None):
    kernel = get_kernel(kernel_path)
    return lambda x,y : distance(x.type(TensorD),deconv_func(y,kernel))


def compare(grad_func, iters_list, model, img, ytrue, crit, ysecond, SoftMax=None, more=1):
    ret = Tensor(2,2, len(iters_list))
    yhat = model(img)
    _,b = yhat.topk(2)
    ysecond = TensorL(1).fill_(b[0,1].data[0])
    root_dir = 'compare/'
    for i, iter in enumerate(iters_list):
        prefix = root_dir +  str(iter)
        grad, new_img = grad_func(model, img, ytrue, crit, ysecond, SoftMax, more, iter)
        #new_img = Variable(img.data.clone() + grad.data)
        y = model(new_img)
        print('iter: ', iter)
        print(y.topk(2))
        a,b = y.topk(2)
        ret[:,0,i] = a.data
        ret[:, 1, i] = b.data
        save_image(new_img.data,  prefix + '_new.jpg', normalize=True)
        save_image(new_img.data - img.data,prefix + '_diff_new_org.jpg', normalize=True)
        save_image(-new_img.data + img.data,prefix + '_diff_org_new.jpg', normalize=True)
        region_window_size1 = [9, 11]#, 13, 15, 17, 19, 21, 23, 25]
        region_window_size2 = [9, 11]#, 13, 15, 17, 19, 21, 23, 25]
        all_windows = dict()
        all_windows['orig'] = img
        for k in region_window_size1:
            for l in region_window_size2:
                convolver = nn.Sequential(nn.Conv2d(1,1,kernel_size=(l, k), bias=False))
                if cuda:
                    convolver = convolver.cuda()
                region_new_org, applied = get_regions(convolver, new_img - img, img, 5000,True)
                all_windows[str(iter) + '_' +str(k) + 'x' + str(l)] = applied
                all_windows[str(iter) + '_' +str(k) + 'x' + str(l) + 'r'] = region_new_org.clone()
                #region_new_org[region_new_org > 0] = 1
                mp = region_new_org.sum(dim=0).sum(dim=0)
                mp[mp>0] = 1
                print(mp)
                print(mp.sum())
                region_new_org += mp
                region_new_org = region_new_org.clamp(max=1)
                all_windows[str(iter) + '_w' + str(k) + 'x' + str(l) + 'r'] = region_new_org
                all_windows[str(iter) + '_s' + str(k) + 'x' + str(l) + 'r'] = tensor_to_img_range(region_new_org) + tensor_to_img_range(img.data) / 3
        save_images(all_windows,prefix+'_all.jpg', 0, 2, iter)
        save_image(applied, prefix + '_regions.jpg', normalize=True)
    return ret


#imgs: pair title, image
def save_images(imgs, path, first, second, iter, cnt_second=0):
    cnt = len(imgs) + cnt_second
    tot = cnt // 12
    k = 0
    mydpi = 144
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())
    rows = 3 #(cnt // 4) + 1
    img_plot_id = 1
    for img_name, img_data in imgs.items():
        cur_img = img_data.clone() if len(img_data.size()) == 3 else img_data.clone().squeeze(0)
        cur_img = cur_img.permute(1, 2, 0)
        a = fig.add_subplot(rows, 4, img_plot_id)
        if 'Var' in str(type(cur_img)):
            cur_img = cur_img.data
        cur_img = tensor_to_img_range(cur_img)
        #if cuda:
        #print('type, ', type(cur_img.data))
        if cuda:
            cur_img = cur_img.cpu()
        #print('type, ', type(cur_img.data))
        #print('type, ', type(cur_img))
        plt.imshow(cur_img.numpy())
        plt.title(img_name)
        #fl_name = path + str(first) + '_' + str(second) + '_' + img_name + '_' + str(iter) + '.jpg'
        #print(fl_name)
        #misc.imsave(fl_name, cur_img.numpy())
        img_plot_id += 1
        if img_plot_id == 13:
            fl_name = path + str(first) + '_' + str(second)  + '_' + str(iter) + '_' + str(
                k) + '.jpg'
            plt.savefig(fl_name, dpi=mydpi)
            plt.close()
            fig = plt.figure()
            plt.axis('off')
            #mng = plt.get_current_fig_manager()
            #mng.resize(*mng.window.maxsize())
            img_plot_id = 1
            k += 1
        #if img_plot_id == 4:
        #    img_plot_id = 5
    fl_name = path + str(first) + '_' + str(second)  + '_' + str(iter) + '_' + str(
        k) + '.jpg'
    plt.savefig(fl_name, dpi=mydpi)
    plt.close()
    return None


cuda, Tensor, TensorL, TensorD, TensorB = is_cuda()
cross_ent = nn.CrossEntropyLoss()
if cuda:
    cross_ent = cross_ent.cuda()
