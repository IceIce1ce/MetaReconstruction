import glob
import numpy as np
import scipy.io

folder1 = len(glob.glob('dataset/ped2/testing/frames/01/*jpg'))
folder2 = len(glob.glob('dataset/ped2/testing/frames/02/*jpg'))
folder3 = len(glob.glob('dataset/ped2/testing/frames/03/*jpg'))
folder4 = len(glob.glob('dataset/ped2/testing/frames/04/*jpg'))
folder5 = len(glob.glob('dataset/ped2/testing/frames/05/*jpg'))
folder6 = len(glob.glob('dataset/ped2/testing/frames/06/*jpg'))
folder7 = len(glob.glob('dataset/ped2/testing/frames/07/*jpg'))
folder8 = len(glob.glob('dataset/ped2/testing/frames/08/*jpg'))
folder9 = len(glob.glob('dataset/ped2/testing/frames/09/*jpg'))
folder10 = len(glob.glob('dataset/ped2/testing/frames/10/*jpg'))
folder11 = len(glob.glob('dataset/ped2/testing/frames/11/*jpg'))
folder12 = len(glob.glob('dataset/ped2/testing/frames/12/*jpg'))

mask1 = np.zeros(folder1, dtype=int)
mask2 = np.zeros(folder2, dtype=int)
mask3 = np.zeros(folder3, dtype=int)
mask4 = np.zeros(folder4, dtype=int)
mask5 = np.zeros(folder5, dtype=int)
mask6 = np.zeros(folder6, dtype=int)
mask7 = np.zeros(folder7, dtype=int)
mask8 = np.zeros(folder8, dtype=int)
mask9 = np.zeros(folder9, dtype=int)
mask10 = np.zeros(folder10, dtype=int)
mask11 = np.zeros(folder11, dtype=int)
mask12 = np.zeros(folder12, dtype=int)

gt_list = []
gt_path = scipy.io.loadmat('dataset/ped2/ped2.mat')['gt'][0]
for gt_tuple in gt_path:
    gt_tuple = gt_tuple.squeeze()
    start, end = gt_tuple[0], gt_tuple[1]
    gt_list.append([start, end])

mask1[gt_list[0][0]:gt_list[0][1]] = 1
mask2[gt_list[1][0]:gt_list[1][1]] = 1
mask3[gt_list[2][0]:gt_list[2][1]] = 1
mask4[gt_list[3][0]:gt_list[3][1]] = 1
mask5[gt_list[4][0]:gt_list[4][1]] = 1
mask6[gt_list[5][0]:gt_list[5][1]] = 1
mask7[gt_list[6][0]:gt_list[6][1]] = 1
mask8[gt_list[7][0]:gt_list[7][1]] = 1
mask9[gt_list[8][0]:gt_list[8][1]] = 1
mask10[gt_list[9][0]:gt_list[9][1]] = 1
mask11[gt_list[10][0]:gt_list[10][1]] = 1
mask12[gt_list[11][0]:gt_list[11][1]] = 1
result = np.concatenate((mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10, mask11, mask12))
np.save('labels/frame_labels_ped2.npy', np.array([result]))