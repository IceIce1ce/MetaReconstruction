import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from build_dataloader import Reconstruction3DDataLoader, DataLoaderMNAD
from models.autoencoder.autoencoder import convAE
from models.memae.memae_3dconv import AutoEncoderCov3DMem
from models.memae.entropy_loss import EntropyLossEncap
import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import cv2
from utils import psnr, anomaly_score_list, AUC, point_score, score_sum, anomaly_score_list_inv
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import argparse
import time
from torch.nn import functional as F
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description="AE evaluation")
### General config for all models ###
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--vid_dir', type=str, default=None, help='save video frames file')
parser.add_argument('--print_time', action='store_true', help='print forward time')
parser.add_argument('--model_name', type=str, default='3dae', choices=['3dae', 'steal', 'memae3d', 'r_mnad', 'p_mnad'], help='name of model')
### Config for MemAE model ###
parser.add_argument('--channel_in_memae', type=int, default=1)
parser.add_argument('--mem_dim', type=int, default=2000)
parser.add_argument('--shrink_thres', type=float, default=0.0025)
parser.add_argument('--entropy_loss_weight', type=float, default=0.0002)
### Config for MNAD model ###
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--m_items_dir', type=str, default='exp/ped2/keys_00.pt', help='directory of model')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
args = parser.parse_args()

if args.vid_dir is not None:
    if not os.path.exists(args.vid_dir):
        os.makedirs(args.vid_dir)

if args.model_name == '3dae' or args.model_name == 'steal':
    loss_func_mse = nn.MSELoss(reduction='none')
    model = convAE()
elif args.model_name == 'memae3d':
    loss_func_mse = nn.MSELoss(reduction='mean')
    loss_func_entropy = EntropyLossEncap()
    model = AutoEncoderCov3DMem(args.channel_in_memae, args.mem_dim, args.shrink_thres)
elif args.model_name == 'r_mnad' or args.model_name == 'p_mnad':
    loss_func_mse = nn.MSELoss(reduction='none')
    model = torch.load(args.model_dir)
    m_items = torch.load(args.m_items_dir)
if args.model_name == '3dae' or args.model_name == 'steal' or args.model_name == 'memae3d':
    model = nn.DataParallel(model)
    model_dict = torch.load(args.model_dir)
    try:
        model_weight = model_dict['model']
        model.load_state_dict(model_weight.state_dict())
    except KeyError:
        model.load_state_dict(model_dict['model_statedict'])
model.cuda()
labels = np.load('./labels/frame_labels_' + args.dataset_type + '.npy')
test_folder = os.path.join(args.dataset_path, args.dataset_type, 'testing', 'frames')
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
if args.model_name == '3dae' or args.model_name == 'steal':
    test_dataset = Reconstruction3DDataLoader(test_folder, transforms.Compose([transforms.ToTensor()]),
                                              resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers_test, drop_last=False)
elif args.model_name == 'memae3d':
    if args.channel_in_memae == 1:
        norm_mean, norm_std = [0.5], [0.5]
    elif args.channel_in_memae == 3:
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    test_dataset = Reconstruction3DDataLoader(test_folder, transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]),
                                              resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers_test, drop_last=False)
elif args.model_name == 'r_mnad' or args.model_name == 'p_mnad':
    test_dataset = DataLoaderMNAD(test_folder, transforms.Compose([transforms.ToTensor()]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers_test, drop_last=False)

videos = OrderedDict()
if args.model_name == '3dae' or args.model_name == 'steal' or args.model_name == 'memae3d':
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*/')))
    for video in videos_list:
        video_name = video.split('/')[-2]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])
elif args.model_name == 'r_mnad' or args.model_name == 'p_mnad':
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
### For MNAD model ###
feature_distance_list = {}

print('Evaluation of', args.dataset_type)
for video in sorted(videos_list):
    if args.model_name == '3dae' or args.model_name == 'steal' or args.model_name == 'memae3d':
        video_name = video.split('/')[-2]
        labels_list = np.append(labels_list, labels[0][8 + label_length:videos[video_name]['length'] + label_length - 7])
    elif args.model_name == 'r_mnad':
        video_name = video.split('/')[-1]
        labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length'] + label_length])
    elif args.model_name == 'p_mnad':
        video_name = video.split('/')[-1]
        labels_list = np.append(labels_list, labels[0][4 + label_length:videos[video_name]['length'] + label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    ### For MNAD model ###
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
if args.model_name == '3dae' or args.model_name == 'steal' or args.model_name == 'memae3d':
    label_length += videos[videos_list[video_num].split('/')[-2]]['length']
elif args.model_name == 'r_mnad' or args.model_name == 'p_mnad':
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    m_items_test = m_items.clone()

model.eval()
tic = time.time()
for k, imgs in enumerate(test_batch):
    if args.model_name == '3dae' or args.model_name == 'steal' or args.model_name == 'memae3d':
        if k == label_length - 15 * (video_num + 1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-2]]['length']
    elif args.model_name == 'r_mnad':
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    elif args.model_name == 'p_mnad':
        if k == label_length - 4 * (video_num + 1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    imgs = Variable(imgs).cuda()
    with torch.no_grad():
        if args.model_name == '3dae' or args.model_name == 'steal':
            outputs = model(imgs)
            loss_mse = loss_func_mse(outputs[0, :, 8], imgs[0, :, 8])
        elif args.model_name == 'memae3d':
            outputs = model(imgs)['output']
            att_w = model(imgs)['att']
            loss_mse = loss_func_mse(outputs[0, :, 8], imgs[0, :, 8])
            loss_entropy = loss_func_entropy(att_w)
            loss_mse = loss_mse + args.entropy_loss_weight * loss_entropy
        elif args.model_name == 'r_mnad':
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
            mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)).item()
            mse_feas = compactness_loss.item()
            point_sc = point_score(outputs, imgs)  # Eq 6 and Eq 7
            if point_sc < args.th:  # higher than threshold -> abnormal frame -> not use it for updating memory items
                query = F.normalize(feas, dim=1)
                query = query.permute(0, 2, 3, 1)
                m_items_test = model.module.memory.update(query, m_items_test, False)
        elif args.model_name == 'p_mnad':
            outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:, 0:3 * 4], m_items_test, False)
            mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (imgs[0, 3 * 4:] + 1) / 2)).item()
            mse_feas = compactness_loss.item()
            point_sc = point_score(outputs, imgs[:, 3 * 4:])
            if point_sc < args.th:
                query = F.normalize(feas, dim=1)
                query = query.permute(0, 2, 3, 1)
                m_items_test = model.module.memory.update(query, m_items_test, False)
    if args.model_name == '3dae' or args.model_name == 'steal' or args.model_name == 'memae3d':
        loss_pixel = torch.mean(loss_mse)
        mse_imgs = loss_pixel.item()
        psnr_list[videos_list[video_num].split('/')[-2]].append(psnr(mse_imgs))
    elif args.model_name == 'r_mnad' or args.model_name == 'p_mnad':
        psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
        feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)

    ### For visualization ###
    if args.vid_dir is not None:
        # save reconstruction error
        output = (outputs[0, :, 8].cpu().detach().numpy() + 1) * 127.5
        output = output.transpose(1, 2, 0).astype(dtype=np.uint8) # [256, 256, 1]
        cv2.imwrite(os.path.join(args.vid_dir, 'out_{:04d}.png').format(k), output)
        # save original image
        saveimgs = (imgs[0, :, 8].cpu().detach().numpy() + 1) * 127.5
        saveimgs = saveimgs.transpose(1, 2, 0).astype(dtype=np.uint8)
        cv2.imwrite(os.path.join(args.vid_dir, 'GT_{:04d}.png').format(k), saveimgs)
        # save color map of reconstruction error
        mseimgs = loss_func_mse(outputs[0, :, 8], imgs[0, :, 8])[0].cpu().detach().numpy()
        mseimgs = mseimgs[:, :, np.newaxis]
        mseimgs = (mseimgs - np.min(mseimgs)) / (np.max(mseimgs) - np.min(mseimgs))
        mseimgs = mseimgs * 255
        mseimgs = mseimgs.astype(dtype=np.uint8)
        color_mseimgs = cv2.applyColorMap(mseimgs, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args.vid_dir, 'MSE_{:04d}.png').format(k), color_mseimgs)

toc = time.time()
if args.print_time:
    time_elapsed = (toc-tic)/len(test_batch)
    print('Processing time:', time_elapsed)
    print('FPS:', 1/time_elapsed)

# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
vid_idx = []
for vi, video in enumerate(sorted(videos_list)):
    if args.model_name == '3dae' or args.model_name == 'steal' or args.model_name == 'memae3d':
        video_name = video.split('/')[-2]
        score = anomaly_score_list(psnr_list[video_name])
        anomaly_score_total_list += score
        vid_idx += [vi for _ in range(len(score))]
    elif args.model_name == 'r_mnad' or args.model_name == 'p_mnad':
        video_name = video.split('/')[-1]
        score = score_sum(anomaly_score_list(psnr_list[video_name]), anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)
        anomaly_score_total_list += score
        vid_idx += [vi for _ in range(len(score))]

anomaly_score_total_list = np.asarray(anomaly_score_total_list)
accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))
print('AUC: ', accuracy * 100.0, '%')

### For visualization ###
if args.vid_dir is not None:
    a = 0
    vids_len = []
    while a < len(vid_idx):
        start_a = a
        cur_vid_idx = vid_idx[a]
        num_frames = 0
        while vid_idx[a] == cur_vid_idx:
            num_frames += 1
            a += 1
            if a >= len(vid_idx):
                break
        vids_len.append(num_frames)
    a = 0
    while a < len(vid_idx):
        start_a = a
        atemp = a
        cur_vid_idx = vid_idx[a]
        vid_len = vids_len[cur_vid_idx]
        # rectangle position
        idx = 0
        rect_start = []
        rect_end = []
        anom_status = False
        while vid_idx[atemp] == cur_vid_idx:
            if not anom_status:
                if labels_list[atemp] == 1:
                    anom_status = True
                    rect_start.append(idx)
            else:
                if labels_list[atemp] == 0:
                    anom_status = False
                    rect_end.append(idx)
            idx += 1
            atemp += 1
            if atemp >= len(vid_idx):
                break
        if anom_status:
            rect_end.append(idx - 1)
        while vid_idx[a] == cur_vid_idx:
            imggt = cv2.imread(os.path.join(args.vid_dir, 'GT_{:04d}.png').format(a))[:, :, [2,1,0]]
            plt.axis('off')
            plt.subplot(231)
            plt.title('Frame', fontsize='small')
            plt.imshow(imggt)

            imgout = cv2.imread(os.path.join(args.vid_dir, 'out_{:04d}.png').format(a))[:,:,[2,1,0]]
            plt.axis('off')
            plt.subplot(232)
            plt.title('Reconstruction', fontsize='small')
            plt.axis('off')
            plt.imshow(imgout)

            imgmse = mpimg.imread(os.path.join(args.vid_dir, 'MSE_{:04d}.png').format(a))
            plt.subplot(233)
            plt.title('Reconstruction Error', fontsize='small')
            plt.axis('off')
            plt.imshow(imgmse)

            plt.subplot(212)
            plt.plot(range(a-start_a+1), 1-anomaly_score_total_list[start_a:a+1], label='prediction', color='blue')
            plt.xlim(0, vid_len-1)
            plt.xticks(fontsize='x-small')
            plt.xlabel('Frames', fontsize='x-small')
            plt.ylim(-0.01, 1.01)
            plt.ylabel('Anomaly Score', fontsize='x-small')
            plt.yticks(fontsize='x-small')
            plt.title('Anomaly Score Over Time')
            for rs, re in zip(rect_start, rect_end):
                currentAxis = plt.gca()
                currentAxis.add_patch(Rectangle((rs, -0.01), re-rs, 1.02, facecolor="pink"))
            plt.savefig(os.path.join(args.vid_dir, 'frame_{:02d}_{:04d}.png').format(cur_vid_idx, a-start_a), dpi=300)
            plt.close()

            a += 1
            if a >= len(vid_idx):
                break