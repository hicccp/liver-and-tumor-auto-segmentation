# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import measure
import scipy.ndimage as nd
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical
from glob import glob


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    # for key, val in param.items():
    #     log_file.write(key + ':' + str(val) + '\n')
    log_file.write(str(param))
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou

def get_dice(pred, gt):
    total_dice = 0.0
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(1.0+torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
        print(dice)
        total_dice += dice

    return total_dice

def get_mc_dice(pred, gt, num=2):
    # num is the total number of classes, include the background
    total_dice = np.zeros(num-1)
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        for j in range(1, num):
            pred_tmp = (pred[i]==j)
            gt_tmp = (gt[i]==j)
            dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(1.0+torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
            total_dice[j-1] +=dice
    return total_dice

def post_processing(prediction):
    prediction = nd.binary_fill_holes(prediction)
    label_cc, num_cc = measure.label(prediction,return_num=True)
    total_cc = np.sum(prediction)
    measure.regionprops(label_cc)
    for cc in range(1,num_cc+1):
        single_cc = (label_cc==cc)
        single_vol = np.sum(single_cc)
        if single_vol/total_cc<0.2:
            prediction[single_cc]=0

    return prediction


# +
def resample_image3D(
    image3D,
    newspacing=[0.3,0.3,3],
    newsize=None,
    method='Linear',):
    """做插值"""
    resample = sitk.ResampleImageFilter()
    if method == 'Linear':
        resample.SetInterpolator(sitk.sitkLinear)
    elif method == 'Nearest':
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputDirection(image3D.GetDirection())
    resample.SetOutputOrigin(image3D.GetOrigin())
    resample.SetOutputSpacing(newspacing)

    if not newsize:
        newsize = np.round(np.array(image3D.GetSize())*np.abs(image3D.GetSpacing())/np.array(newspacing)).astype('int').tolist()
    resample.SetSize(newsize)

    newimage = resample.Execute(image3D)
    return newimage

def sitk_onehot_transform(image):
    image_array = sitk.GetArrayFromImage(image)
    label_array_onehot = to_categorical(image_array)
    image_onehot = sitk.GetImageFromArray(label_array_onehot)
    image_onehot.SetOrigin(image.GetOrigin())
    image_onehot.SetDirection(image.GetDirection())
    image_onehot.SetSpacing(image.GetSpacing())
    return image_onehot

def make_out_itk(image,image_sitk):
    out_image_sitk = sitk.GetImageFromArray(image)
    out_image_sitk.SetSpacing(image_sitk.GetSpacing())
    out_image_sitk.SetOrigin(image_sitk.GetOrigin())
    out_image_sitk.SetDirection(image_sitk.GetDirection())
    return out_image_sitk

# 数组替换元素
def array_replace(array,olds,news):
    # 不适用于onehot
    #olds:list of old value
    #news:list of new value
    olds = np.array(olds)
    news = np.array(news)
    offset = olds.max()*10
    tmps = olds+offset
    array += offset
    for tmp,new in zip(tmps,news):
        array[array==tmp] = new
    return array


# -
def plot_slice_sample(image,label,d,fn=None):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image[:,:,d].squeeze())
    a.set_title('image')
    plt.colorbar(orientation='horizontal')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(label[:,:,d].squeeze())
    imgplot.set_clim(0.0, 3.0)
    a.set_title('label')
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    if fn is not None:
        plt.savefig(fn)
    plt.show()


def plot_slice_sample(image,label,d,cmap='gray',fn=None):
    # find contour
    contours = measure.find_contours(label[:,:,d], level=0)
    
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image[:,:,d].squeeze())#,cmap=cmap
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, linestyle='-', color='red')
    ax.set_title('image')
    plt.colorbar(orientation='horizontal')
    
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(label[:,:,d].squeeze())#,cmap=cmap
    #imgplot.set_clim(0.0, 3.0)
    ax.set_title('label')
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    
    if fn is not None:
        plt.savefig(fn)
    plt.show()


def remove_files(re):
    """读取所有匹配re规则的文件并删除"""
    listt = glob(re)
    for fn in listt:
        os.remove(fn)
    return listt


from skimage import measure
def filter_connected_domain(image,min_region_area=None,num_keep_region=None,ratio_keep=None):
    """
    原文链接：https://blog.csdn.net/a563562675/article/details/107066836
    return label of filter 
    """
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, background=0, return_num=True)
    if num < 1:
        return image

    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(0, num)]
    area_list = [region[i].area for i in num_list]
    
    # 去除面积较小的连通域
    if min_region_area:
        drop_list = np.where(np.array(area_list)<min_region_area)[0]
        for i in drop_list:
            label[region[i].slice][region[i].image] = 0
    elif ratio_keep:
        max_region_area = np.array(area_list).max()
        drop_list = np.where(np.array(area_list)<max_region_area*ratio_keep)[0]
        for i in drop_list:
            label[region[i].slice][region[i].image] = 0 
    
    else:
        if len(num_list) > num_keep_region:
            num_list_sorted = sorted(num_list, key=lambda x: area_list[x])[::-1]# 面积由大到小排序
            for i in num_list_sorted[num_keep_region:]:
                # label[label==i] = 0
                label[region[i].slice][region[i].image] = 0
#             num_list_sorted = num_list_sorted[:num_keep_region]
    return label


def get_relabel_from_seg(label_sitk,seg_sitk):
    """
    功能：实践发现：部分病例的Segmentation-label.nrrd文件不正确，缺失其中一个或者多个类别。
    因此，以Segmentation.seg.nrrd为依据，重建Segmentation-label.nrrd
    输入：
        label_sitk: 从原Segmentation-label.nrrd文件读取的Image对象
        seg_sitk： 从原Segmentation.seg.nrrd文件读取的Image对象
    输出：
        重建后的label_sitk的Image对象
    """
    # get array
    label = sitk.GetArrayFromImage(label_sitk).transpose((2,1,0))
    seg = sitk.GetArrayFromImage(seg_sitk).transpose((2,1,0,3))
    assert len(seg.shape)==4
    
    # get offset
    h_RIEO,w_RIEO,d_RIEO = [int(item) for item in seg_sitk.GetMetaData("Segmentation_ReferenceImageExtentOffset").split()]
    # get seg shape
    h_seg,w_seg,d_seg = seg_sitk.GetSize()
    # initial full_seg
    full_seg = np.zeros(list(label.shape)+[1+seg.shape[-1]])
    # get full_seg
    full_seg[ h_RIEO:h_seg+h_RIEO, w_RIEO:w_seg+w_RIEO, d_RIEO:d_seg+d_RIEO, 1:] = seg
    full_seg[ h_RIEO:h_seg+h_RIEO, w_RIEO:w_seg+w_RIEO, d_RIEO:d_seg+d_RIEO, 0] = 1-seg.sum(axis=-1)
    full_seg = full_seg.astype(np.uint8)
    
    # get re_label
    re_label = np.argmax(full_seg,axis=-1).astype(np.uint8)
    
    # get re_label_sitk
    re_label_sitk = sitk.GetImageFromArray(re_label.transpose(2,1,0))
    re_label_sitk.CopyInformation(label_sitk)
    
    return re_label_sitk
