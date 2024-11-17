import numpy as np
from math import fabs
import math
import torch
import os
import logging
import h5py
import torchvision.transforms.functional as TF
import random
from PIL import Image
import torch.nn.functional as F
import cv2
# from timers import Timer, CudaTimer

def check_event_nums(h5_file_lists, args):
    """
    当按照固定事件数据将event转换为tensor的时候，需要保证event的个数，大于：
    int(num_events_per_pixel * DVS_stream_width * DVS_stream_height) + 1
    否则，判定该event序列无效。
    当按照 从零累积灰度事件 并 固定事件数据将event转换为tensor的时候，需要保证event的个数，大于：
    int(num_events_per_pixel * DVS_stream_width * DVS_stream_height) + 1 + pre_acc_num(提前累积的事件个数)
    否则，判定该event序列无效。
    """
    if args.train_divide_events_by_frames:
        return h5_file_lists
    valid_files = []
    for each_file in h5_file_lists:
        with h5py.File(each_file, 'r') as f:
            num_events = f['events']['xs'].shape[0]
            DVS_stream_height, DVS_stream_width = f.attrs['DVS_sensor_height'], f.attrs['DVS_sensor_width']
            if args.window_size is not None:
                window_events_size = args.window_events_size
            else:
                window_events_size = int(args.num_events_per_pixel * DVS_stream_width * DVS_stream_height)
            if args.acc4zero_flag or args.acc4zero_divide_merge:
                if args.pre_acc_ratio_train_dynamic:
                    min_nums = window_events_size * args.unrolling_len + \
                               args.pre_acc_ratio_train_end * DVS_stream_width * DVS_stream_height
                else:
                    min_nums = window_events_size * args.unrolling_len + \
                               args.pre_acc_ratio_train * DVS_stream_width * DVS_stream_height

            else:
                min_nums = window_events_size * args.unrolling_len
            if num_events > min_nums + 100:
                valid_files.append(each_file)
    return valid_files





def image_proess(inp_img,inp_event,tar_img,ps,opt):


    w, h = tar_img.shape[1], tar_img.shape[2]
    padw = ps - w if w < ps else 0
    padh = ps - h if h < ps else 0

    # Reflect Pad in case image is smaller than patch_size
    if padw != 0 or padh != 0:
        inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
        tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
        inp_event = TF.pad(inp_event, (0, 0, padw, padh), padding_mode='reflect')



    inp_img = torch.from_numpy(inp_img)
    tar_img = torch.from_numpy(tar_img)
    inp_event = torch.from_numpy(inp_event)

    hh, ww = tar_img.shape[1], tar_img.shape[2]

    rr = random.randint(0, hh - ps)
    cc = random.randint(0, ww - ps)
    aug = random.randint(0, 8)

    # Crop patch
    inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
    tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]
    inp_event = inp_event[:, rr:rr + ps, cc:cc + ps]

    # Data Augmentations
    if aug == 1:
        inp_img = inp_img.flip(1)
        tar_img = tar_img.flip(1)
        inp_event = inp_event.flip(1)

    elif aug == 2:
        inp_img = inp_img.flip(2)
        tar_img = tar_img.flip(2)
        inp_event = inp_event.flip(2)

    elif aug == 3:
        inp_img = torch.rot90(inp_img, dims=(1, 2))
        tar_img = torch.rot90(tar_img, dims=(1, 2))
        inp_event = torch.rot90(inp_event, dims=(1, 2))

    elif aug == 4:
        inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
        tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        inp_event = torch.rot90(inp_event, dims=(1, 2), k=2)

    elif aug == 5:
        inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
        tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        inp_event = torch.rot90(inp_event, dims=(1, 2), k=3)

    elif aug == 6:
        inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
        tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        inp_event = torch.rot90(inp_event.flip(1), dims=(1, 2))

    elif aug == 7:
        inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
        tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
        inp_event = torch.rot90(inp_event.flip(2), dims=(1, 2))


    input_img=inp_img
    input_event=inp_event
    target=tar_img
    return input_img,input_event,target



def image_proess_val(inp_img,inp_event,tar_img,opt):
    img_multiple_of=8


    inp_img = torch.from_numpy(inp_img).unsqueeze(0)
    tar_img = torch.from_numpy(tar_img).unsqueeze(0)
    inp_event = torch.from_numpy(inp_event).unsqueeze(0)



    h,w = inp_img.shape[2], inp_img.shape[3]
    H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-h if h%img_multiple_of!=0 else 0
    padw = W-w if w%img_multiple_of!=0 else 0
    inp_img = F.pad(inp_img, (0,padw,0,padh), 'reflect')
    tar_img = F.pad(tar_img, (0,padw,0,padh), 'reflect')
    inp_event = F.pad(inp_event, (0,padw,0,padh), 'reflect')


    input_img=inp_img
    input_event=inp_event
    target=tar_img
    return input_img, input_event,target

def image_proess_test(inp_img,input_acc,input_div,opt):
    img_multiple_of=8
    inp_img = Image.fromarray(inp_img)
    input_acc = Image.fromarray(input_acc)
    input_div = Image.fromarray(input_div)




    inp_img = TF.to_tensor(inp_img).unsqueeze(0)
    input_div = TF.to_tensor(input_div).unsqueeze(0)
    input_acc = TF.to_tensor(input_acc).unsqueeze(0)


    h,w = inp_img.shape[2], inp_img.shape[3]
    H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-h if h%img_multiple_of!=0 else 0
    padw = W-w if w%img_multiple_of!=0 else 0
    inp_img = F.pad(inp_img, (0,padw,0,padh), 'reflect')
    input_div = F.pad(input_div, (0,padw,0,padh), 'reflect')
    input_acc = F.pad(input_acc, (0,padw,0,padh), 'reflect')


    input=torch.cat((inp_img,input_acc,input_div), dim=1)
    return input




def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def make_gray_event_preview(events, num_bin_to_show=0):
    # events: [1 x C x H x W] gray-event torch-tensor

    # Grayscale mode
    # normalize event image to [0, 255] for display
    events_preview = events[0, num_bin_to_show, :, :].detach().cpu().numpy()
    min_val = np.min(events_preview)
    max_val = np.max(events_preview)
    events_preview = np.clip((255.0 * (events_preview - min_val)/(max_val - min_val)).astype(np.uint8), 0, 255)
    return events_preview



def make_binary_event_preview(events, mode='red-blue', num_bins_to_show=-1):
    # events: [1 x C x H x W] binary-event torch-tensor
    # mode: 'red-blue' or 'grayscale'
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    assert(mode in ['red-blue', 'grayscale'])
    if num_bins_to_show < 0:
        sum_events = torch.sum(events[0, :, :, :], dim=0).detach().cpu().numpy()

    else:
        sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0).detach().cpu().numpy()

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[sum_events > 0] = 255
        r[sum_events < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -10.0, 10.0
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    return event_preview




cuda_timers = {}
timers = {}


class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end))

class EventPreprocessor:
    """
    # ma:　receive the torch tensor as input. 归一化采用的减均值除方差。接受的 event 尺寸为： [batch, C, H, W]
    Utility class to preprocess event tensors.
    Can perform operations such as hot pixel removing, event tensor normalization,
    or flipping the event tensor.

    -- 202104201006
    这个函数好像不太对啊，这个函数的操作是对一个 batchsize 中所有的样本混合在一起，对非零值做 减均值-除方差 归一化。
    但是，按道理来讲，这种归一化，应该是对每个单独的样本做吧，而不是将所有样本混合在一起。
    因此，修改这个归一化的代码。
    """

    def __init__(self, options):
        logging.info('== Event preprocessing ==')
        self.no_normalize = options.no_normalize
        if self.no_normalize:
            logging.info('!!Will not normalize event tensors!!')
        else:
            logging.info('Will normalize event tensors.')

        self.hot_pixel_locations = []
        if options.hot_pixels_file:
            try:
                self.hot_pixel_locations = np.loadtxt(options.hot_pixels_file, delimiter=',').astype(int)
                logging.info('Will remove {} hot pixels'.format(self.hot_pixel_locations.shape[0]))
            except IOError:
                logging.info('WARNING: could not load hot pixels file: {}'.format(options.hot_pixels_file))

        self.flip = options.flip
        if self.flip:
            logging.info('Will flip event tensors.')

        self.args = options

    def __call__(self, events):
        """
        接受的 event 尺寸为： [batch, C, H, W]
        """
        # receive the torch tensor as input

        # Remove (i.e. zero out) the hot pixels
        for x, y in self.hot_pixel_locations:
            events[:, :, y, x] = 0

        # Flip tensor vertically and horizontally
        if self.flip:
            events = torch.flip(events, dims=[2, 3])

        # Normalize the event tensor (voxel grid) so that
        # the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
        if not self.no_normalize:  # 减均值除方差 对一个batchsize中的所有sample进行
            with CudaTimer('Normalization'):
                nonzero_ev = (events != 0)
                num_nonzeros = nonzero_ev.sum()
                if num_nonzeros > 0:
                    if self.args.norm_method == 'normal':
                        # compute mean and stddev of the **nonzero** elements of the event tensor
                        # we do not use PyTorch's default mean() and std() functions since it's faster
                        # to compute it by hand than applying those funcs to a masked array
                        mean = events.sum() / num_nonzeros
                        stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                        mask = nonzero_ev.float()
                        events = mask * (events - mean) / stddev
                    elif self.args.norm_method == 'minmax':
                        max_val = torch.max(events)
                        min_val = torch.min(events)
                        mask = nonzero_ev.float()
                        events = mask * (events - min_val) / max_val
                    elif self.args.norm_method == 'max':
                        max_val = torch.max(events)
                        mask = nonzero_ev.float()
                        events = mask * events / max_val


        return events