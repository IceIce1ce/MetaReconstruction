import torch.utils.data as data
import torchvision.transforms as transforms
from build_dataloader import Reconstruction3DDataLoader, Reconstruction3DDataLoaderJump, DataLoaderMNAD
from models.autoencoder.autoencoder import convAE
from models.memae.memae_3dconv import AutoEncoderCov3DMem
from models.memae.memae_3dconv import weights_init_memae
from models.memae.entropy_loss import EntropyLossEncap
from models.mnad.r_mnad import r_mnad
from models.mnad.p_mnad import p_mnad
import argparse
import os
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description="AE training")
### General config for all models ###
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2', 'avenue', 'shanghai'], help='type of dataset')
parser.add_argument('--dataset_path', type=str, default='dataset', help='directory of data')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='adam or sgd with momentum and cosine annealing lr')
parser.add_argument('--model_dir', type=str, default=None, help='path of model for resume')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')
parser.add_argument('--model_name', type=str, default='3dae', choices=['3dae', 'steal', 'memae3d', 'r_mnad', 'p_mnad'], help='name of model')
### Config for steal model ###
parser.add_argument('--pseudo_anomaly_jump', type=float, default=0.01, help='pseudo anomaly jump frame, 0 for no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[2, 3, 4, 5], help='Jump for pseudo anomaly')
### Config for memae3d model ###
parser.add_argument('--channel_in_memae', type=int, default=1)
parser.add_argument('--mem_dim', type=int, default=2000)
parser.add_argument('--shrink_thres', type=float, default=0.0025)
parser.add_argument('--entropy_loss_weight', type=float, default=0.0002)
### Config for mnad model ###
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
args = parser.parse_args()

train_folder = os.path.join(args.dataset_path, args.dataset_type, 'training', 'frames')
img_extension = '.tif' if args.dataset_type == 'ped1' else '.jpg'
### Dataloader for 3D AE model ###
if args.model_name == '3dae' or args.model_name == 'steal':
    train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor()]), resize_height=args.h,
                                               resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    ### Dataloader for STEAL model ###
    train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]), resize_height=args.h, resize_width=args.w,
                                                        dataset=args.dataset_type, img_extension=img_extension, jump=args.jump)
    train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
### Dataloader for MemAE3d model ###
elif args.model_name == 'memae3d':
    if args.channel_in_memae == 1:
        norm_mean, norm_std = [0.5], [0.5]
    elif args.channel_in_memae == 3:
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    train_dataset = Reconstruction3DDataLoader(train_folder, transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]),
                                               resize_height=args.h, resize_width=args.w, dataset=args.dataset_type, img_extension=img_extension)
    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
### Dataloader for MNAD model ###
elif args.model_name == 'r_mnad' or args.model_name == 'p_mnad':
    train_dataset = DataLoaderMNAD(train_folder, transforms.Compose([transforms.ToTensor()]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)
    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
log_dir = os.path.join('./exp', args.dataset_type)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print('Train ' + args.model_name + ' model')
if args.start_epoch < args.epochs:
    if args.model_name == '3dae' or args.model_name == 'steal':
        model = convAE()
    elif args.model_name == 'memae3d':
        model = AutoEncoderCov3DMem(1, args.mem_dim, args.shrink_thres)
        model.apply(weights_init_memae) # init weights to avoid NaN loss
    elif args.model_name == 'r_mnad':
        model = r_mnad(memory_size=args.msize, feature_dim=args.fdim, key_dim=args.mdim)
    elif args.model_name == 'p_mnad':
        model = p_mnad(t_length=args.t_length, memory_size=args.msize, feature_dim=args.fdim, key_dim=args.mdim)
    model = nn.DataParallel(model)
    model.cuda()
    if args.model_name == '3dae' or args.model_name == 'steal' or args.model_name == 'memae3d':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model_name == 'r_mnad' or args.model_name == 'p_mnad':
        params_encoder = list(model.module.encoder.parameters())  # use module for data parallel setting
        params_decoder = list(model.module.decoder.parameters())
        optimizer = torch.optim.Adam(params_encoder + params_decoder, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # init memory items
    if args.model_dir is not None:
        assert args.start_epoch > 0
        model_dict = torch.load(args.model_dir)
        model_weight = model_dict['model']
        model.load_state_dict(model_weight.state_dict())
        optimizer.load_state_dict(model_dict['optimizer'])
        model.cuda()
    for epoch in range(args.start_epoch, args.epochs):
        if args.model_name == '3dae':
            loss_func_mse = nn.MSELoss(reduction='mean')
            total_loss = 0
            for j, imgs in enumerate(train_batch):
                net_in = Variable(imgs).cuda()
                outputs = model(net_in)
                loss = loss_func_mse(outputs, net_in)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print('Epoch:', epoch)
            print('Loss:', total_loss)
            model_dict = {'model': model, 'optimizer': optimizer.state_dict()}
            torch.save(model_dict, os.path.join(log_dir, args.model_name + '_{:02d}.pth'.format(epoch)))
        elif args.model_name == 'steal':
            loss_func_mse = nn.MSELoss(reduction='none')
            lossepoch, losscounter, pseudolossepoch, pseudolosscounter = 0, 0, 0, 0
            for j, (imgs, imgsjump) in enumerate(zip(train_batch, train_batch_jump)):
                net_in = Variable(imgs).cuda()
                jump_pseudo_stat = []
                for b in range(args.batch_size):
                    total_pseudo_prob = 0
                    rand_number = np.random.rand()
                    # skip frame pseudo anomaly
                    pseudo_anomaly_jump = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly_jump
                    total_pseudo_prob += args.pseudo_anomaly_jump
                    if pseudo_anomaly_jump:
                        net_in[b] = imgsjump[b] # update for pseudo locations
                        jump_pseudo_stat.append(True)
                    else:
                        jump_pseudo_stat.append(False)
                outputs = model(net_in)
                loss_mse = loss_func_mse(outputs, net_in)
                modified_loss_mse = []
                for b in range(args.batch_size):
                    if jump_pseudo_stat[b]:
                        modified_loss_mse.append(torch.mean(-loss_mse[b]))
                        pseudolossepoch += modified_loss_mse[-1].cpu().detach().item()
                        pseudolosscounter += 1
                    else: # no pseudo anomaly
                        modified_loss_mse.append(torch.mean(loss_mse[b]))
                        lossepoch += modified_loss_mse[-1].cpu().detach().item()
                        losscounter += 1
                assert len(modified_loss_mse) == loss_mse.size(0)
                stacked_loss_mse = torch.stack(modified_loss_mse)
                loss = torch.mean(stacked_loss_mse)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Epoch:', epoch)
            if pseudolosscounter != 0:
                print('PseudoMeanLoss: {:.9f}'.format(pseudolossepoch / pseudolosscounter))
            if losscounter != 0:
                print('MeanLoss: {:.9f}'.format(lossepoch / losscounter))
            model_dict = {'model': model, 'optimizer': optimizer.state_dict()}
            torch.save(model_dict, os.path.join(log_dir, args.model_name + '_{:02d}.pth'.format(epoch)))
        elif args.model_name == 'memae3d':
            loss_func_mse = nn.MSELoss(reduction='mean')
            loss_func_entropy = EntropyLossEncap()
            total_loss = 0
            for j, imgs in enumerate(train_batch):
                net_in = Variable(imgs).cuda()
                outputs = model(net_in)['output']
                att_w = model(net_in)['att']
                loss = loss_func_mse(outputs, net_in)
                entropy_loss = loss_func_entropy(att_w)
                loss = loss + args.entropy_loss_weight * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print('Epoch:', epoch)
            print('Loss:', total_loss)
            model_dict = {'model': model, 'optimizer': optimizer.state_dict()}
            torch.save(model_dict, os.path.join(log_dir, args.model_name + '_{:02d}.pth'.format(epoch)))
        elif args.model_name == 'r_mnad':
            loss_func_mse = nn.MSELoss(reduction='none')
            total_loss = 0.0
            total_separateness_loss = 0.0
            total_compactness_loss = 0.0
            model.train()
            for j, imgs in enumerate(train_batch):
                net_in = Variable(imgs).cuda()
                outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(net_in, m_items, True)
                optimizer.zero_grad()
                loss_pixel = torch.mean(loss_func_mse(outputs, net_in))
                loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss += loss.item()
                total_separateness_loss += separateness_loss.item()
                total_compactness_loss += compactness_loss.item()
            scheduler.step()
            print('Epoch:', epoch)
            print('Loss: {:.3f}/ Separateness loss {:.3f}/ Compactness loss {:.3f}'.format(total_loss, total_separateness_loss, total_compactness_loss))
            torch.save(model, os.path.join(log_dir, args.model_name + '_{:02d}.pth'.format(epoch)))
            torch.save(m_items, os.path.join(log_dir, 'keys' + '_{:02d}.pt'.format(epoch)))
        elif args.model_name == 'p_mnad':
            loss_func_mse = nn.MSELoss(reduction='none')
            total_loss = 0.0
            total_separateness_loss = 0.0
            total_compactness_loss = 0.0
            model.train()
            for j, imgs in enumerate(train_batch):
                net_in = Variable(imgs).cuda()
                outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(net_in[:, 0:12], m_items, True)
                optimizer.zero_grad()
                loss_pixel = torch.mean(loss_func_mse(outputs, net_in[:, 12:]))
                loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss += loss.item()
                total_separateness_loss += separateness_loss.item()
                total_compactness_loss += compactness_loss.item()
            scheduler.step()
            print('Epoch:', epoch)
            print('Loss: {:.3f}/ Separateness loss {:.3f}/ Compactness loss {:.3f}'.format(total_loss, total_separateness_loss, total_compactness_loss))
            torch.save(model, os.path.join(log_dir, args.model_name + '_{:02d}.pth'.format(epoch)))
            torch.save(m_items, os.path.join(log_dir, 'keys' + '_{:02d}.pt'.format(epoch)))