import torch
import torch.nn.functional as F
import torch.optim

import numpy as np
import os
import shutil

from utils import sample_gt, softmax, softmax_new
from datasets import get_dataset, get_originate_dataset
from skimage.segmentation import felzenszwalb
import time
from scipy.stats import entropy

def get_pseudo_label(segments, TRAIN_Y, gt):
    MAX_S = np.max(segments)
    MAX_Y = np.max(TRAIN_Y)
    pseudo_label = np.zeros([np.shape(TRAIN_Y)[0], np.shape(TRAIN_Y)[1], TRAIN_Y.max() + 1])
    idx = TRAIN_Y > 0
    tmp_Y, tmp_s = TRAIN_Y[idx], segments[idx]

    for i_tmp_s, i_tmp_Y in zip(tmp_s, tmp_Y):
        if i_tmp_Y > 0:
            pseudo_label[segments == i_tmp_s, i_tmp_Y] = 1

    pseudo_label[gt == 0, :] = 0
    return pseudo_label


def four_rotation(matrix_0):
    matrix_90 = np.rot90(matrix_0, k=1, axes=(0, 1))
    matrix_180 = np.rot90(matrix_90, k=1, axes=(0, 1))
    matrix_270 = np.rot90(matrix_180, k=1, axes=(0, 1))
    return [matrix_0, matrix_90, matrix_180, matrix_270]


def rotation(matrix_x, matrix_y, pseudo_label=None, segments=None, Mirror=False):
    train_PL, train_SG = [], []
    if pseudo_label is None: train_PL = None
    if segments is None: train_SG = None
    if Mirror == True:
        train_IMG, train_Y = four_rotation(matrix_x[::-1, :, :]), four_rotation(matrix_y[::-1, :])
        if pseudo_label is not None:
            for k_pseudo_label in pseudo_label:
                train_PL.append(four_rotation(k_pseudo_label[::-1, :, :]))
        if segments is not None:
            for k_segments in segments:
                train_SG.append(four_rotation(k_segments[::-1, :, :]))
    else:
        train_IMG, train_Y = four_rotation(matrix_x), four_rotation(matrix_y)
        if pseudo_label is not None:
            for k_pseudo_label in pseudo_label:
                train_PL.append(four_rotation(k_pseudo_label))
        if segments is not None:
            for k_segments in segments:
                train_SG.append(four_rotation(k_segments))

    return train_IMG, train_Y, train_PL, train_SG


def H_segment(img, train_gt, params, gt):
    pseudo_label, idx = [], []
    path = "Datasets/" + params.DATASET + '/' + params.DATASET + '_felzenszwalb.npy'

    if os.path.exists(path) == True:
        all_segment = np.load("Datasets/" + params.DATASET + '/' + params.DATASET + '_felzenszwalb.npy')
    else:
        idd, idy = 1, []
        while idd < np.shape(gt)[0] * np.shape(gt)[1]:
            current_segment = felzenszwalb(img, scale=1.0, sigma=0.95, min_size=idd)
            if len(idy) > 0:
                if np.sum(current_segment - idy[-1]) != 0:
                    print(idd, np.sum(current_segment - idy[-1]), len(idy))
                    idy.append(current_segment)
            else:
                idy.append(current_segment)
            _, counts = np.unique(current_segment, return_counts=True)
            idd = max(counts.min(), idd + 1)

        all_segment = np.stack(idx, 0)
        np.save(path, all_segment)

    for current_segment in all_segment:
        tmp = get_pseudo_label(current_segment, train_gt, gt)
        count_tmp = tmp.sum(-1)
        if np.sum(count_tmp > 0) == np.sum(gt > 0):
            print(len(pseudo_label))
            return pseudo_label

        if len(idx) > 0:
            for k_idx in idx:
                tmp[k_idx > 0, :] = 0

        if tmp.sum() > 0:
            print(len(pseudo_label), count_tmp.max(), np.sum(count_tmp > 0), np.sum(gt > 0))
            idx.append(count_tmp)
            pseudo_label.append(tmp)


def pre_data(img, train_gt, params, gt,pseudo_labels3):
    start = time.time()
    TRAIN_IMG, TRAIN_Y, TRAIN_PL, TRAIN_SG = [], [], [], []
    #pseudo_labels = H_segment(img, train_gt, params, gt)
    #pseudo_labels2=[pseudo_labels[0]]
    pseudo_labels2 = [pseudo_labels3]
    ##
    # pseudo_labels=None
    ##
    train_IMG, train_Y, train_PL, train_SG = rotation(img, train_gt, pseudo_labels2)
    train_IMG_M, train_Y_M, train_PL_M, train_SG_M = rotation(img, train_gt, pseudo_labels2, Mirror=True)
    print(time.time() - start)
    image_Column = torch.Tensor(np.stack((train_IMG[0], train_IMG[2], train_IMG_M[0], train_IMG_M[2]), 0))#.permute(0, 3,
                                                                                                          #         1, 2)
    y_Column = torch.LongTensor(np.stack((train_Y[0], train_Y[2], train_Y_M[0], train_Y_M[2]), 0).astype(int))

    image_Row = torch.Tensor(np.stack((train_IMG[1], train_IMG[3], train_IMG_M[1], train_IMG_M[3]), 0))#.permute(0, 3, 1,
                                                                                                       #         2)
    y_Row = torch.LongTensor(np.stack((train_Y[1], train_Y[3], train_Y_M[1], train_Y_M[3]), 0).astype(int))
    print(time.time() - start)
    if train_PL is not None:
        y_PL_Column, y_PL_Row = [], []
        for k_PL, k_PL_M in zip(train_PL, train_PL_M):
            y_PL_Column.append(torch.FloatTensor(np.stack((k_PL[0], k_PL[2], k_PL_M[0], k_PL_M[2]), 0).astype(float)))
            y_PL_Row.append(torch.FloatTensor(np.stack((k_PL[1], k_PL[3], k_PL_M[1], k_PL_M[3]), 0).astype(float)))
        TRAIN_PL.append(y_PL_Column)
        TRAIN_PL.append(y_PL_Row)
        print(time.time() - start)
    else:
        TRAIN_PL = None

    if train_SG is not None:
        y_SG_Column, y_SG_Row = [], []
        for k_SG, k_SG_M in zip(train_SG, train_SG_M):
            y_SG_Column.append(torch.FloatTensor(np.stack((k_SG[0], k_SG[2], k_SG_M[0], k_SG_M[2]), 0).astype(float)))
            y_SG_Row.append(torch.FloatTensor(np.stack((k_SG[1], k_SG[3], k_SG_M[1], k_SG_M[3]), 0).astype(float)))
        TRAIN_SG.append(y_SG_Column)
        TRAIN_SG.append(y_SG_Row)
        print(time.time() - start)
    else:
        TRAIN_SG = None

    TRAIN_IMG.append(image_Column)
    TRAIN_IMG.append(image_Row)
    TRAIN_Y.append(y_Column)
    TRAIN_Y.append(y_Row)
    print(time.time() - start)
    return TRAIN_IMG, TRAIN_Y, TRAIN_PL, TRAIN_SG

def sample_gt3_new(img, gt, train_gt, test_gt, SAMPLE_PERCENTAGE, IGNORED_LABELS, ALL_LABELS):
    X = img
    Y = gt
    labels = np.unique(Y)
    labels = np.array([val for val in labels if not val in IGNORED_LABELS])
    print(labels)
    row, col, n_band = X.shape
    num_class = len(ALL_LABELS) # TODO: Fix to the correct amount
    max_class = np.max(ALL_LABELS)
    # num_class = len(np.unique(labels))
    first = True
    skipped_labels = []
    for i, val in enumerate(ALL_LABELS): # range(1, num_class + 1):
        if val in labels:
            index = np.where(train_gt == val)
            index2 = np.where(test_gt == val)
            if first:
                array1_train = index[0]
                array2_train = index[1]
                array1_test = index2[0]
                array2_test = index2[1]
                first = False
            else:
                array1_train = np.concatenate((array1_train, index[0]))
                array2_train = np.concatenate((array2_train, index[1]))
                array1_test = np.concatenate((array1_test, index2[0]))
                array2_test = np.concatenate((array2_test, index2[1]))
        else:
            skipped_labels.append(val)
    y_train = Y[array1_train, array2_train]
    trueEDtimesSID = []
    trueEDtimesSID2 = []
    sumtrueES = []
    sumfalseES = []
    pseudo_labels3 = np.zeros([row, col, num_class])
    for i in range(0, len(array1_test)):
        if i % 1000 == 0:
            print("i:%d" % (i))
        #if i%200!=0:
        #    continue
        xtest = array1_test[i]
        ytest = array2_test[i]
        labeltest = Y[xtest, ytest]
        specvectortest = X[xtest, ytest]
        # i
        EDs = np.zeros(num_class)
        SIDs = np.zeros(num_class)
        EDtimesSIDs = np.zeros(num_class)
        EDtimesSIDs2 = np.zeros(num_class)
        minED = 10000000000
        for j, val in enumerate(ALL_LABELS): #range(1, num_class + 1):  # 类别循环
            if val in labels:
                index2 = np.where(y_train == val)  ## 当前类别序号
                index2 = index2[0]
                EDsclass = []
                SIDclass = []
                EDtimesSIDclass = []
                for nn in range(0, len(index2)):  # 类别内训练集循环 nn
                    # print(index2[nn])##当前训练样本序号
                    ind = index2[nn]
                    xtrain = array1_train[ind]
                    ytrain = array2_train[ind]
                    specvectortrain = X[xtrain, ytrain]
                    ED = np.sqrt(np.square(xtest - xtrain) + np.square(ytest - ytrain))
                    SID1 = entropy(specvectortest, specvectortrain)
                    SID2 = entropy(specvectortrain, specvectortest)
                    SID = SID1 + SID2
                    EDtimesSID = np.sqrt(ED * SID)
                    ED = ED + SID
                    EDsclass.append(ED)
                    SIDclass.append(SID)
                    EDtimesSIDclass.append(EDtimesSID)

                    if ED < minED:
                        minED = ED
            # =================================
                inde = np.argsort(EDsclass)

                jiaquan = 0
                for nn in range(0, len(index2)):
                    jiaquandis = EDsclass[inde[nn]] * (float(num_class) ** (-nn))  # 类别内训练集循环 nn
                    jiaquan = jiaquan + jiaquandis

                EDs[j] = jiaquan
                SIDs[j] = np.min(SIDclass)
                EDtimesSIDs[j] = np.min(EDtimesSIDclass)
                jiaquan2 = 0
                inde2 = np.argsort(EDtimesSIDclass)
                for nn in range(0, len(index2)):
                    jiaquandis = EDtimesSIDclass[inde2[nn]] * (float(num_class) ** (-nn))  # 类别内训练集循环 nn
                    jiaquan2 = jiaquan2 + jiaquandis
                EDtimesSIDs2[j] = jiaquan2
            else:
                EDs[j] = np.nan
                SIDs[j] = np.nan
                EDtimesSIDs[j] = np.nan
                EDtimesSIDs2[j] = np.nan
        if np.nanmin(EDtimesSIDs) > 0.085:
            continue
        else:
            minn = np.nanmin(EDs)
            softm3 = softmax_new(-EDtimesSIDs2 * max_class*100)
            minn
        try:
            labeEDtimesSIDs = np.nanargmin(EDtimesSIDs)
            labeEDtimesSIDs2 = np.nanargmin(EDtimesSIDs2)
        except ValueError:
            labeEDtimesSIDs = np.argmin(EDtimesSIDs)
            labeEDtimesSIDs2 = np.argmin(EDtimesSIDs2)
        train_gt[xtest, ytest] = labeEDtimesSIDs2
        pseudo_labels3[xtest, ytest][ALL_LABELS] = softm3


        if labeEDtimesSIDs == labeltest:
            trueEDtimesSID.append(1)
            sumtrueES.append(np.nanmin(EDtimesSIDs))
        else:
            trueEDtimesSID.append(0)
            sumfalseES.append(np.nanmin(EDtimesSIDs))
            print("falseEDtimesSID:", np.nanmin(EDtimesSIDs))

        if labeEDtimesSIDs2 == labeltest:
            trueEDtimesSID2.append(1)
        else:
            trueEDtimesSID2.append(0)
    accuEDtimesSID2 = np.sum(trueEDtimesSID2) / len(trueEDtimesSID2)

    print("lenEDtimesSID2: %d,accurate:%f, truenum:%d" % (
    len(trueEDtimesSID2), 100 * accuEDtimesSID2, np.sum(trueEDtimesSID2)))
    return train_gt,pseudo_labels3

def sample_gt3(img, gt, train_gt, test_gt, SAMPLE_PERCENTAGE):
    X = img
    Y = gt
    row, col, n_band = X.shape
    num_class = np.max(Y)
    for i in range(1, num_class + 1):
        index = np.where(train_gt == i)
        index2 = np.where(test_gt == i)

        if i == 1:
            array1_train = index[0]
            array2_train = index[1]
            array1_test = index2[0]
            array2_test = index2[1]
        else:
            array1_train = np.concatenate((array1_train, index[0]))
            array2_train = np.concatenate((array2_train, index[1]))
            array1_test = np.concatenate((array1_test, index2[0]))
            array2_test = np.concatenate((array2_test, index2[1]))
    y_train = Y[array1_train, array2_train]
    trueEDtimesSID = []
    trueEDtimesSID2 = []
    sumtrueES = []
    sumfalseES = []
    pseudo_labels3 = np.zeros([row, col, num_class + 1])
    for i in range(0, len(array1_test)):
        if i % 1000 == 0:
            print("i:%d" % (i))
        #if i%200!=0:
        #    continue
        xtest = array1_test[i]
        ytest = array2_test[i]
        labeltest = Y[xtest, ytest]
        specvectortest = X[xtest, ytest]
        i
        EDs = np.zeros(num_class)
        SIDs = np.zeros(num_class)
        EDtimesSIDs = np.zeros(num_class)
        EDtimesSIDs2 = np.zeros(num_class)
        minED = 10000000000
        for j in range(1, num_class + 1):  # 类别循环
            index2 = np.where(y_train == j)  ## 当前类别序号
            index2 = index2[0]
            EDsclass = []
            SIDclass = []
            EDtimesSIDclass = []
            for nn in range(0, len(index2)):  # 类别内训练集循环 nn
                # print(index2[nn])##当前训练样本序号
                ind = index2[nn]
                xtrain = array1_train[ind]
                ytrain = array2_train[ind]
                specvectortrain = X[xtrain, ytrain]
                ED = np.sqrt(np.square(xtest - xtrain) + np.square(ytest - ytrain))
                SID1 = entropy(specvectortest, specvectortrain)
                SID2 = entropy(specvectortrain, specvectortest)
                SID = SID1 + SID2
                EDtimesSID = np.sqrt(ED * SID)
                ED = ED + SID
                EDsclass.append(ED)
                SIDclass.append(SID)
                EDtimesSIDclass.append(EDtimesSID)

                if ED < minED:
                    minED = ED
            # =================================
            inde = np.argsort(EDsclass)

            jiaquan = 0
            for nn in range(0, len(index2)):
                jiaquandis = EDsclass[inde[nn]] * (float(num_class) ** (-nn))  # 类别内训练集循环 nn
                jiaquan = jiaquan + jiaquandis

            EDs[j - 1] = jiaquan
            SIDs[j - 1] = np.min(SIDclass)
            EDtimesSIDs[j - 1] = np.min(EDtimesSIDclass)
            ###
            jiaquan2 = 0
            inde2 = np.argsort(EDtimesSIDclass)
            for nn in range(0, len(index2)):
                jiaquandis = EDtimesSIDclass[inde2[nn]] * (float(num_class) ** (-nn))  # 类别内训练集循环 nn
                jiaquan2 = jiaquan2 + jiaquandis
            EDtimesSIDs2[j - 1] = jiaquan2
        ###
        # ========================
        # print("minED:", minED)
        # if minED>2.71:
        if np.min(EDtimesSIDs) > 0.085:
            continue
        else:
            minn = np.min(EDs)
            softm = softmax(16 / EDs)
            softm2 = softmax(-EDs * num_class)
            softm3 = softmax(-EDtimesSIDs2 * num_class*100)
            minn
        
        labeEDtimesSIDs = np.argmin(EDtimesSIDs) + 1
        labeEDtimesSIDs2 = np.argmin(EDtimesSIDs2) + 1
        train_gt[xtest, ytest] = labeEDtimesSIDs2
        pseudo_labels3[xtest, ytest][1:17] = softm3


        if labeEDtimesSIDs == labeltest:
            trueEDtimesSID.append(1)
            sumtrueES.append(np.min(EDtimesSIDs))
        else:
            trueEDtimesSID.append(0)
            sumfalseES.append(np.min(EDtimesSIDs))
            # print("falseEDtimesSID:", np.min(EDtimesSIDs))

        if labeEDtimesSIDs2 == labeltest:
            trueEDtimesSID2.append(1)
        else:
            trueEDtimesSID2.append(0)
    accuEDtimesSID2 = np.sum(trueEDtimesSID2) / len(trueEDtimesSID2)

    print("lenEDtimesSID2: %d,accurate:%f, truenum:%d" % (
    len(trueEDtimesSID2), 100 * accuEDtimesSID2, np.sum(trueEDtimesSID2)))
    return train_gt,pseudo_labels3


def single_pad(pad_value):
        if pad_value%2 == 0:
            return [int(pad_value/2), int(pad_value/2)]
        else:
            return [int(pad_value/2), int(pad_value/2) + 1]


def calc_pad(mx, shape, axes):
    pad_values = mx - shape
    pads = [[0,0] for _ in shape]
    for axis in axes:
        #print(i)
        #print(i in axes)
        pads[axis] = (single_pad(pad_values[axis]))
    # print(pads)
    return pads
    

def concat_with_padding(list_of_arrays, padding_axes):
    shape1s = [i.shape for i in list_of_arrays]
    max_shape = np.max(shape1s, axis=0)
    #max_shape = np.array(img[-1].shape)
    pads = [calc_pad(max_shape, i, padding_axes) for i in shape1s]
    img2 = [np.pad(list_of_arrays[i], pad) for i, pad in enumerate(pads)]
    res2 = np.concatenate(img2)
    return res2


def load_datasets(DATASET, datasets_root, SAMPLE_PERCENTAGE):
    img, gt, LABEL_VALUES, IGNORED_LABELS, ALL_LABELS, _, _ = get_dataset(DATASET, datasets_root)
    X, Y = get_originate_dataset(DATASET, datasets_root)
    img = concat_with_padding(img, [1])
    img = img[:, :, list(range(0, 102, 3))] # Why is this here? This takes the every third channel from every dataset.
                                            # The end point being fixed at 102 might be a problem. 
                                            # On the other hand no similar thing is done to X, which is supposedly
                                            # the same image. 

    N_CLASSES = len(LABEL_VALUES)
    INPUT_SIZE = np.shape(img)[-1]
    # train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode='fixed')
    train_test_gt = [sample_gt(i, SAMPLE_PERCENTAGE, mode='fixed') for i in gt]
    #train_gt = sample_gt2(X, Y, train_gt, test_gt, SAMPLE_PERCENTAGE)
    # # # breakpoint()
    pseudo_labelpath = str(DATASET) + f'/pseudo_labels/pseudo_labels3/pseudo_labels3_{SAMPLE_PERCENTAGE}.npy'
    pseudo_labels3 = []
    if not os.path.exists(pseudo_labelpath):
        newdir = str(DATASET) + '/pseudo_labels/pseudo_labels3/'
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        for x, y, tr_te_gt in zip(X, Y, train_test_gt):
            pseudo_labels3.append(sample_gt3_new(x, y, tr_te_gt[0], tr_te_gt[1], 
                                                 SAMPLE_PERCENTAGE, IGNORED_LABELS, ALL_LABELS)[1])
        pseudo_labels3 = concat_with_padding(pseudo_labels3, [1])
        np.save(pseudo_labelpath, pseudo_labels3)
    else:
        pseudo_labels3=np.load(pseudo_labelpath)
    train_gt = concat_with_padding([i[0] for i in train_test_gt], [1])
    test_gt = concat_with_padding([i[1] for i in train_test_gt], [1])
    gt = concat_with_padding(gt, [1]) 
    X = concat_with_padding(X, [1])
    Y = concat_with_padding(Y, [1])
    return img, gt, LABEL_VALUES, IGNORED_LABELS, ALL_LABELS, X, Y, N_CLASSES,\
           INPUT_SIZE, train_gt, test_gt, pseudo_labels3

def move_file(old_path, new_path, old_filename, new_filename):
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    old_path = old_path + old_filename
    new_path = new_path + new_filename
    shutil.move(old_path, new_path)
    
def save_loss_acc(path, name_root, losses, accuracies):
    if not os.path.exists(path):
        os.makedirs(path)
    new_name_acc = path + name_root +  '_accuracy.npy'
    new_name_loss = path + name_root + '_losses.npy'
    np.save(new_name_acc, accuracies)
    np.save(new_name_loss, losses)
    
def save_states(path, states, fname):
    if not os.path.exists(path):
        os.makedirs(path)
    save_file_path = path + fname
    torch.save(states, save_file_path)

    

def end_of_training_saves(temp_path, best_path, latest_path, temp_name, latest_name,
                          best_name_root, states, losses, accuracies):
    save_states(latest_path, states, latest_name)
    move_file(temp_path, best_path, temp_name, best_name_root + '.pth')
    save_loss_acc(best_path, best_name_root, losses, accuracies)
    shutil.rmtree(temp_path)

def mid_training_saves(temp_path, temp_name_root, states, losses, accuracies):
    save_states(temp_path, states, temp_name_root + '.pth')
    save_loss_acc(path=temp_path, name_root=temp_name_root, losses=losses, accuracies=accuracies)
    