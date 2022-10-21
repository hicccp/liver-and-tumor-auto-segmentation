# -*- coding: utf-8 -*-
# +
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from skimage import transform
from tqdm import tqdm
import torch.nn.functional as F
import skimage

import sys
sys.path.append('../')
import networks


# -

def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


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


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, mode='max', best_filepath='best_model.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        if mode=='max':# larger is better for score
            self.val_max = -np.Inf
        elif mode=='min':# smaller is better for score
            self.val_min = np.Inf
        self.delta = delta
        self.mode = mode
        self.best_filepath = best_filepath

    def __call__(self, score, model):
        # larger is better for score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        
        elif self.mode=='min' and score >= self.best_score-self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode=='max' and score <= self.best_score+self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.mode=='min':
            if self.verbose:
                print(f'Validation score decreased ({self.val_min:.6f} --> {score:.6f}).  Saving model ...')
            self.val_min = score
        elif self.mode=='max':
            if self.verbose:
                print(f'Validation score increased ({self.val_max:.6f} --> {score:.6f}).  Saving model ...')
            self.val_max = score
        from torch.nn.parallel.data_parallel import DataParallel
        if isinstance(model,DataParallel):
            model = model.module
        torch.save(model, self.best_filepath)                 # 这里会存储迄今最优的模型


# +
import matplotlib.pyplot as plt
from skimage import measure
def draw_contour_fig(image, label_gt, label_pred):
    """
    image: HW
    label_gt: HW
    label_pred: HW
    """
    import matplotlib
    matplotlib.use('Agg')
    # 画图
    contours_gt = measure.find_contours(array=label_gt, level=0)
    contours_pred = measure.find_contours(array=label_pred, level=0)
    h,w = image.shape
    dpi=100
    figsize = (3*h/dpi,3*w/dpi)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.imshow(image,cmap=plt.cm.gray)
    for n, contour in enumerate(contours_gt):
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2, linestyle='dashed')
    for n, contour in enumerate(contours_pred):
        ax.plot(contour[:, 1], contour[:, 0], color='orange', linewidth=2, linestyle='-')
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # 获取图像的数组    
    fig_array = fig2data(fig)
    return fig_array

def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image_array = np.asarray(image)
    
    plt.close()
    return image_array


# -

import SimpleITK as sitk
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


def preprocess_for_inference(image):
    """
    image:numpy array,format HWDC
    """
    # 提取前景,然后剔除黑框
    # 1)自动阈值分割
    from skimage import data,filters
    thresh = filters.threshold_li(image) #返回一个阈值
    dst = (image >= thresh)*1.0   #根据阈值进行分割
    # 2)连通域分析,取出过小的连通域
    filter_mask = filter_connected_domain(dst,min_region_area=None,num_keep_region=None,ratio_keep=0.1)
    filter_mask = (filter_mask>0).astype(np.uint8)
    # 3)剔除纯背景的黑框
    background_value = 0
    is_foreground = filter_mask
    tmp = np.nonzero(is_foreground)
    hmin, hmax = tmp[0].min(), tmp[0].max()
    wmin, wmax = tmp[1].min(), tmp[1].max()
    dmin, dmax = tmp[2].min(), tmp[2].max(),
    print( "shape before crop: ", image.shape )
    image = image[hmin:hmax+1, wmin:wmax+1, dmin:dmax+1]
    print( "shape after crop: ", image.shape )
        
    # 归一化到正负1之间
    # 参考论文《Liver segmentation and metastases detection in MR images using convolutional neural networks》
    lower = np.percentile(image.flatten(),q=0)#参考论文：0 # 第一次实验：5
    upper = np.percentile(image.flatten(),q=99.8)# 参考论文99.8 # 第一次实验95
    image = (image-lower)/(upper-lower)
    image = (image-0.5)*2 # 之前训练的肝脏分割模型妖气执行这一步，但是最新的肿瘤分割模型要求没有这一步。
    
    return image,(hmin, hmax, wmin, wmax, dmin, dmax),(lower,upper)


def get_model(model_path):
    net = torch.load(model_path)
    net.eval()
    return net


def model_inference(model,input_tensor,out_name=None):
    with torch.no_grad():
        out = model(input_tensor)
        if out_name is not None:
            out = out['out']
        mask = F.softmax(out, dim=1)
        mask = torch.argmax(mask, dim=1)
    return mask.cpu().numpy()#, out.cpu().numpy()


def get_predict_mask_for_3Dimage(
    model,
    image,
    out_name=None,
    device=None
):
    """
    image format: HWD
    mask_3D format: HWD
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_3D = []
    for slice in range(image.shape[2]):
        image = image.astype('float32')
        input_tensor = torch.from_numpy(image[:,:,slice][np.newaxis,np.newaxis,:,:]).to(device)
        mask_slice = model_inference(model,input_tensor,out_name=out_name)
        mask_3D += [mask_slice]
    mask_3D = np.concatenate( mask_3D,axis=0 )#DHW
    mask_3D = mask_3D.transpose((1,2,0)) # HWD
    return mask_3D


def get_post_process_liver_mask(mask_liver_3D):
    """
    using morphology methods to process the predicted liver mask.
    mask_liver_3D format: HWD
    mask_liver_3D_open format: HWD
    """
    import skimage
    # 1) morphology open
    mask_liver_3D_open = skimage.morphology.opening( 
        mask_liver_3D,
        skimage.morphology.ball(radius=4,dtype=np.uint8)
    ).astype(mask_liver_3D.dtype)
    # 2) keep only the largest connected-domain
    mask_liver_3D_post = filter_connected_domain(
        mask_liver_3D_open,
        min_region_area=None,
        num_keep_region=1,
        ratio_keep=None)
    mask_liver_3D_post = (mask_liver_3D_post>0).astype(np.uint8)
    
    return mask_liver_3D_post


def get_predict_lesion_mask_for_3Dimage(model_lesion, model_mode, image, mask_liver_3D, out_name=None, pt=0.5):
    """
    get predicted lesion mask with the navigation of liver mask
    
    model_lesion: torch model object
    model_mode: '2D' or '3D'
    image format: HWD
    mask_liver_3D: format: HWD
    out_name: some model_lesion ouputs more than one tensor, out_name specifies the required one
    pt: the probability threshold
    """
    
    # get image volume for lesion mask predicting
#     mask_liver_3D = skimage.morphology.opening(
#         mask_liver_3D,
#         skimage.morphology.ball(radius=5,dtype=np.uint8)
#     ).astype(np.uint8)
    tmp = np.nonzero(mask_liver_3D)
    h,w,d = mask_liver_3D.shape
    hmin,hmax = tmp[0].min(), tmp[0].max()
    wmin,wmax = tmp[1].min(), tmp[1].max()
    dmin,dmax = tmp[2].min(), tmp[2].max()
    hc,wc = int((hmin+hmax)/2), int((wmin+wmax)/2)
    hmin,hmax = max(0,min(hmin,hc-64)), min(h,max(hmax,hc+64))
    wmin,wmax = max(0,min(wmin,wc-64)), min(w,max(wmax,wc+64))
    
    
    
    mask_lesion_3D = np.zeros_like(image).astype(np.uint8)
    probability_lesion_3D = np.zeros_like(image).astype(np.float32)
    if model_mode=='2D':
        from test_util_2D import predict_single_slice, predict_one_slice
        stride_xy = 64
        patch_size = (128,128)
        for slice in range(dmin,dmax+1):
            label_map, score_map = predict_one_slice(#predict_single_slice
                model_lesion, 
                image[hmin:hmax,wmin:wmax,slice]*mask_liver_3D[hmin:hmax,wmin:wmax,slice],
                stride_xy, 
                patch_size, 
                num_classes=2, 
                out_name=out_name,
                device="cuda")
            mask_lesion_3D[hmin:hmax,wmin:wmax,slice] = (score_map[1,:,:].squeeze()>pt).astype(np.uint8)#label_map.astype(np.uint8)
            probability_lesion_3D[hmin:hmax,wmin:wmax,slice] = (score_map[1,:,:].squeeze()).astype(np.float32)
    elif model_mode=='3D':
        from test_util_3D import test_single_case as predict_one_volume
        stride_dhw=(8,48,48)#(16,64,64)
        patch_size=(16,96,96)#(32,128,128)#(16,64,64)
        label_map, score_map = predict_one_volume(
            model_lesion,
            (image[hmin:hmax,wmin:wmax,dmin:dmax]*mask_liver_3D[hmin:hmax,wmin:wmax,dmin:dmax]).transpose((2,0,1)), # HWD to DHW
            stride_dhw=stride_dhw, 
            patch_size=patch_size,
            num_classes=2, 
            device="cuda")
        label_map, score_map = label_map.transpose((1,2,0)), score_map.transpose((0,2,3,1)), # DHW to HWD
        mask_lesion_3D[hmin:hmax,wmin:wmax,dmin:dmax] = (score_map[1,...].squeeze()>pt).astype(np.uint8)#label_map.astype(np.uint8)
        probability_lesion_3D[hmin:hmax,wmin:wmax,dmin:dmax] = (score_map[1,:,:].squeeze()).astype(np.float32)
    mask_lesion_3D = mask_lesion_3D*mask_liver_3D
    probability_lesion_3D = probability_lesion_3D*mask_liver_3D
    return mask_lesion_3D,probability_lesion_3D


# +
def get_post_processed_lesion_mask(
    image_3D,
    proba_lesion_3D,
    mask_liver_3D,
    output_shape=None,
    pt=0.5
):
    """
    return post-processed 3D lesion mask.
    post-processing:
        1) bool operation with liver mask : exclude detected objects(false objects) outside liver
        2) remove small objects(connected domains)
        3) resize
        4) morphology close: to get a round-shape-like object
    """
    
#     # 0) bool
#     mask_liver_3D = skimage.morphology.dilation(
#         mask_liver_3D,
#         skimage.morphology.ball(radius=2,dtype=np.uint8)
#     ).astype(np.uint8)
#     proba_lesion_3D = proba_lesion_3D*mask_liver_3D
    
#     # 1) grabcut
#     mask_lesion_3D = np.zeros_like(proba_lesion_3D).astype(np.uint8)
#     img_lower, img_upper = image_3D.min(),image_3D.max()
#     for slice in range(proba_lesion_3D.shape[2]):
#         # img必须为8位3通道图像
#         img = np.concatenate((image_3D[:,:,[slice]], image_3D[:,:,[slice]], image_3D[:,:,[slice]]), axis=2)
# #         img_lower = 0
# #         img_upper = np.percentile(img,99.8)
#         img = (img-img_lower)/(img_upper-img_lower)*255
#         img = img.astype(np.uint8)
#         proba_mask = proba_lesion_3D[:,:,slice]
#         mask = grabcut(img,proba_mask,thresholds=(0.2,0.5,0.8))
#         mask_lesion_3D[:,:,slice] = mask
#     mask_lesion_3D = (mask_lesion_3D/255.0).astype(np.uint8)    

    mask_lesion_3D = (proba_lesion_3D>pt).astype(np.uint8)# 对于1号测试病例,0.7比较好
    
    # 2) morphology close
    mask_lesion_3D = skimage.morphology.closing( 
        mask_lesion_3D,
        skimage.morphology.ball(radius=2,dtype=np.uint8)
    ).astype(np.uint8)
    
    # 3) remove small objects(connected domains)
    mask_lesion_3D = filter_connected_domain(
        mask_lesion_3D,
        min_region_area=None,
        num_keep_region=3,
        ratio_keep=None)
    mask_lesion_3D = (mask_lesion_3D>0).astype(np.uint8)

    # 4) resize
    if output_shape is not None:
        mask_lesion_3D = transform.resize(
            mask_lesion_3D, # HWD, regard D as color channel
            output_shape=output_shape, 
            order=0, 
            mode='constant',
            cval=0,
            preserve_range=True, 
            anti_aliasing=False,# 实验发现，对于图像，得取值True; 而对于label，得取值False
        ).astype(np.uint8)
    
    
    
    return mask_lesion_3D


# -

def grabcut(img,proba_mask,thresholds=(0.3,0.5,0.8)):
    import cv2
    
    mask = np.zeros_like(proba_mask)
    mask[np.where(proba_mask<=thresholds[0])] = 0 # background
    mask[np.where(proba_mask<=thresholds[1])] = 2 # likely background
    mask[np.where(proba_mask>thresholds[1])] = 3 # likely foreground
    mask[np.where(proba_mask>thresholds[2])] = 1 # foreground
    mask = mask.astype(np.uint8)

    mask_unique = np.unique(mask)
    if not (1 in mask_unique):# or (3 in mask_unique):
        return np.zeros_like(proba_mask)
    
    bgdModel=None
    fgdModel=None
    rect=[0,0,1,1]
    cv2.grabCut(
        img,#[hmin:hmax,wmin:wmax,:], 
        mask,#[hmin:hmax,wmin:wmax], 
        rect, 
        bgdModel, 
        fgdModel, 
        iterCount=10, 
        mode=cv2.GC_INIT_WITH_MASK )

    mask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    return mask


def sitk_onehot_transform(image,num_classes=None,dtype=None):
    from tensorflow.keras.utils import to_categorical
    image_array = sitk.GetArrayFromImage(image)
    if num_classes:
        label_array_onehot = to_categorical(image_array,num_classes)
    else:
        label_array_onehot = to_categorical(image_array)
    if dtype:
        label_array_onehot = label_array_onehot.astype(dtype)
    image_onehot = sitk.GetImageFromArray(label_array_onehot[:,:,:,1:])
    image_onehot.SetOrigin(image.GetOrigin())
    image_onehot.SetDirection(image.GetDirection())
    image_onehot.SetSpacing(image.GetSpacing())
    return image_onehot


# +
def save_mask_as_nrrd(
    mask_3D,
    num_classes,
    image_sitk,
    newsize=None,
    image_filepath=None,
    save_path=None):
    """
    mask_3D: numpy array
    num_classes: number of pixel classes, including background
    image_sitk: sitk obk=ject
    newsize
    """
    
#     if newsize is not None:
#         image_sitk = resample_image3D(image_sitk,newsize=newsize,method='Linear')
        
    import shutil
    Seg_MetaData_dict = {
        'ITK_InputFilterName': 'NrrdImageIO',
        'NRRD_measurement frame': '[UNKNOWN_PRINT_CHARACTERISTICS]\n',
        'Segment0_ID': 'Segment_1',
        'Segment0_Name': 'liver',
        'Segment0_NameAutoGenerated': '0',
        'Segment0_ColorAutoGenerated': '1',

        'Segment1_ID': 'Segment_2',
        'Segment1_Name': 'tumour',
        'Segment1_NameAutoGenerated': '0',
        'Segment1_ColorAutoGenerated': '1',

        'Segmentation_ReferenceImageExtentOffset': '0 0 0',
        'Segmentation_MasterRepresentation': 'Binary labelmap',
        'Segmentation_ContainedRepresentationNames': 'Binary labelmap|'
    }
    
    # getnetrate label_pred_sitk for the 'Segmentation-label.nrrd' file
    label_sitk = sitk.GetImageFromArray(mask_3D.transpose(2,0,1).astype(np.uint8))#transpose order should be check
    label_sitk.CopyInformation(image_sitk) 
    label_sitk = sitk.Cast(label_sitk, sitk.sitkUInt8)
    
    # generate onehot-encoding label (i.e label_pred_onehot_sitk) for the 'Segmentation.seg.nrrd' file
    seg_sitk = sitk_onehot_transform(label_sitk,num_classes=num_classes,dtype='uint8')  
    
    
    # set meta data for seg_sitk
    for key in Seg_MetaData_dict:
        seg_sitk.SetMetaData( key,Seg_MetaData_dict[key] )
    # set Segment%d_Extent
    seg = sitk.GetArrayFromImage(seg_sitk).transpose((2,1,0,3)).astype('uint8')
    for cls in range(num_classes-1):
        tmp = np.where( seg[:,:,:,cls]==1 )
        if len(tmp[0])==0:
            continue
        value = '%d %d %d %d %d %d'%( tmp[0].min(), tmp[0].max(), tmp[1].min(), tmp[1].max(), tmp[2].min(), tmp[2].max() )
        key = 'Segment%d_Extent'%cls
        print(key,value )
        seg_sitk.SetMetaData( key,value ) 
    
    if newsize is not None:
        label_sitk = resample_image3D(label_sitk,newsize=newsize,method='Nearest')
        seg_sitk = resample_image3D(seg_sitk,newsize=newsize,method='Nearest')
        print("Size after resampling: ",seg_sitk.GetSize())
    label_sitk = sitk.Cast(label_sitk, sitk.sitkUInt8)
    
    # save to nrrd file
    if (image_filepath is not None) and (save_path is not None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        label_filepath = os.path.join( save_path,'Segmentation-label.nrrd' ) 
        sitk.WriteImage(label_sitk, label_filepath)
        seg_filepath = os.path.join( save_path,'Segmentation.seg.nrrd' ) 
        sitk.WriteImage(seg_sitk, seg_filepath)
        # copy image nrrd file to save_dir
        source = image_filepath
        target = os.path.join( save_path,os.path.basename(image_filepath) )
        shutil.copy(source, target)
    return label_sitk, seg_sitk


# -

def resample_image3D(
    image3D,
    newspacing=None,
    newsize=None,
    method='Linear',):
    """做插值,以newsize为第一准则，若newsize没有，则以newspacing为第一准则"""
    resample = sitk.ResampleImageFilter()
    if method == 'Linear':
        resample.SetInterpolator(sitk.sitkLinear)
    elif method == 'Nearest':
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputDirection(image3D.GetDirection())
    resample.SetOutputOrigin(image3D.GetOrigin())

    if newsize is not None:
        newspacing = (np.array(image3D.GetSize())*np.abs(image3D.GetSpacing())/np.array(newsize)).tolist()
        resample.SetOutputSpacing(newspacing)
    elif newspacing is not None:
        resample.SetOutputSpacing(newspacing)
        newsize = np.round(np.array(image3D.GetSize())*np.abs(image3D.GetSpacing())/np.array(newspacing)).astype('int').tolist()
    
    resample.SetSize(newsize)
    # resample.SetDefaultPixelValue(0)

    newimage = resample.Execute(image3D)
    return newimage
