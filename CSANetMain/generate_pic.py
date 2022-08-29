import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
from global_module.Utils import extract_samll_cubic
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

def load_dataset():
    # T1_ori = sio.loadmat('...\datasets\Farm1.mat')
    # T2_ori = sio.loadmat('...\datasets\Farm2.mat')
    # mat_gt = sio.loadmat('...\datasets\GTChina1.mat')
    # TT1 = T1_ori['imgh']
    # TT2 = T2_ori['imghl']
    # gt_hsi = mat_gt['label']
    # TOTAL_SIZE = 63000
    # VALIDATION_SPLIT = 0.787
    # TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    T1_ori = sio.loadmat('...\datasets\River_before.mat')
    T2_ori = sio.loadmat('...\datasets\River_after.mat')
    mat_gt = sio.loadmat('...\datasets\Rivergt.mat')
    TT1 = T1_ori['river_before']
    TT2 = T2_ori['river_after']
    gt_hsi = mat_gt['gt']
    TOTAL_SIZE = 111583
    VALIDATION_SPLIT = 0.9664
    TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)


    # T1_ori = sio.loadmat('...\datasets\Sa1.mat')
    # T2_ori = sio.loadmat('...\datasets\Sa2.mat')
    # mat_gt = sio.loadmat('...\datasets\SaGT.mat')
    # TT1 = T1_ori['T1']
    # TT2 = T2_ori['T2']
    # gt_hsi = mat_gt['GT']
    # TOTAL_SIZE = 73987
    # VALIDATION_SPLIT = 0.902
    # TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return TT1, TT2, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT






def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 1:
            y[index] = np.array([255, 255, 255]) / 255.
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
    return y




def generate_iter(TRAIN_SIZE, train_indices,  TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data1, whole_data2, PATCH_LENGTH, padded_data1, padded_data2, INPUT_DIMENSION1,
                  INPUT_DIMENSION2, batch_size, gt):
    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1

    all_data1 = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data1,
                                                       PATCH_LENGTH, padded_data1, INPUT_DIMENSION1)
    all_data2 = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data2,
                                                       PATCH_LENGTH, padded_data2, INPUT_DIMENSION2)

    train_data1 = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data1,
                                                         PATCH_LENGTH, padded_data1, INPUT_DIMENSION1)
    train_data2 = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data2,
                                                         PATCH_LENGTH, padded_data2, INPUT_DIMENSION2)

    x_train1 = train_data1.reshape(train_data1.shape[0], train_data1.shape[1], train_data1.shape[2], INPUT_DIMENSION1)

    x_train2 = train_data2.reshape(train_data2.shape[0], train_data2.shape[1], train_data2.shape[2], INPUT_DIMENSION2)
    all_data1.reshape(all_data1.shape[0], all_data1.shape[1], all_data1.shape[2], INPUT_DIMENSION1)
    all_data2.reshape(all_data2.shape[0], all_data2.shape[1], all_data2.shape[2], INPUT_DIMENSION2)
    x_val1 = all_data1[7000:9000]
    x_val2 = all_data2[7000:9000]
    y_val = gt_all[7000:9000]
    # x_test1 = x_test_all1[:-VAL_SIZE]
    # x_test2 = x_test_all2[:-VAL_SIZE]
    # y_test = y_test[:-VAL_SIZE]
    # print('y_train', np.unique(y_train))
    # print(y1_train)
    # y1_train = to_categorical(y1_train)  # to one-hot labels

    x1_tensor_train = torch.from_numpy(x_train1).type(torch.FloatTensor)
    x2_tensor_train = torch.from_numpy(x_train2).type(torch.FloatTensor)
    y_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, x2_tensor_train, y_tensor_train)


    x1_tensor_valida = torch.from_numpy(x_val1).type(torch.FloatTensor)
    x2_tensor_valida = torch.from_numpy(x_val2).type(torch.FloatTensor)
    y_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, x2_tensor_valida, y_tensor_valida)
    #


    all_tensor_data1 = torch.from_numpy(all_data1).type(torch.FloatTensor)
    all_tensor_data2 = torch.from_numpy(all_data2).type(torch.FloatTensor)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data1, all_tensor_data2, all_tensor_data_label)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valiada_iter = Data.DataLoader(
         dataset=torch_dataset_valida,
         batch_size=batch_size,
         shuffle=True,
         num_workers=0,
     )

    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_iter,  valiada_iter, all_iter  # , y_test


def generate_png(pred_test, gt_hsi, Dataset, total_indices):

    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)

    x_label[total_indices] = pred_test
    x = np.ravel(x_label)

    # print('-------Save the result in mat format--------')
    # x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
    # sio.savemat('mat/' + Dataset + '_' + '.mat', {Dataset: x_re})

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    name = 'CSANetMain'
    path = '../' + name
    classification_map(y_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_' + name + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_gt.png')
    print('------Get classification maps successful-------')

