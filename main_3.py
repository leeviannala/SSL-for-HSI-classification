import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim

import numpy as np
import os
import argparse

from utils import metrics, calprecision
# from DenseConv import *
from DenseConv import DHCN
import argparse
import time
from torch.utils.data import DataLoader, TensorDataset
from functions import pre_data, load_datasets
from functions import mid_training_saves
from functions import make_dirs, how_many_runs
from functions import end_of_training_time, make_ds, permute_batch, make_ds2
from functions import permute_list_items

def presets(params):
    RUNS = 1
    MX_ITER = 1000000000
    SAMPLE_PERCENTAGE = params.SAMPLE_PERCENTAGE
    DATASET = params.DATASET
    DHCN_LAYERS = params.DHCN_LAYERS
    CONV_SIZE = params.CONV_SIZE
    H_MIRROR = params.H_MIRROR
    USE_HARD_LABELS = params.USE_HARD_LABELS
    LR = 1e-3
    os.environ["CUDA_VISIBLE_DEVICES"] = params.GPU

    device = torch.device("cuda:0")
    print(torch.cuda.device_count())

    file_and_folder_name_common_part = '_'.join(
                    [str(params.USE_HARD_LABELS), str(params.SAMPLE_PERCENTAGE), str(params.DHCN_LAYERS), 
                     str(params.CONV_SIZE), str(params.ROT), str(params.MIRROR), str(params.H_MIRROR)])
    temp_path = str(DATASET) + '/tmp' + str(params.GPU) + '_abc/'
    latest_path = str(DATASET) + '/Latest/' + file_and_folder_name_common_part + '/'
    best_path = str(DATASET) + '/Best/' + file_and_folder_name_common_part + '/'

    # TODO: F
    make_dirs([best_path, temp_path, latest_path])
    
    RUNS = how_many_runs(best_path, RUNS)
    run_numbers = np.array(range(RUNS)) + len(os.listdir(best_path))

    return RUNS, MX_ITER, SAMPLE_PERCENTAGE, DATASET, DHCN_LAYERS, CONV_SIZE, \
        H_MIRROR, USE_HARD_LABELS, LR, device, file_and_folder_name_common_part, \
            temp_path, latest_path, best_path, run_numbers 

def pre_run(run, pre):
    return None

def run(run_numbers, pre):
    for r in run_numbers:
        pre_r = pre_run(r, pre)

def train():
    return None

def test():
    return None

def loss_calc():
    return None

def epoch():
    # epoch pretests
    if epoch % 100 == 0:
        print("epoch: %d" % (epoch))
    current_time = time.time()
    if current_time - start_time > 3600:
        
        print(f'Training end due to current_time={current_time-start_time}')
        end_of_training_time(
            model_DHCN, train_gt, test_gt, 
            file_and_folder_name_common_part, best_ACC, tmp_epoch,
            temp_path, best_path, latest_path, losses, accuracies, run)
        break

    if reload_model == True:

        if str(tmp_epoch) in recode_reload:

            recode_reload[str(tmp_epoch)] += 1
            tmp_rate = tmp_rate * 0.1
            if tmp_rate < 1e-6:

                print(f'Training end due to tmp_rate={tmp_rate}')
                end_of_training_time(
                    model_DHCN, train_gt, test_gt, 
                    file_and_folder_name_common_part, best_ACC, tmp_epoch,
                    temp_path, best_path, latest_path, losses, accuracies, run)
                break

            print('learning decay: ', str(tmp_epoch), tmp_rate)
            for param_group in optimizer_DHCN.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1

        else:
            recode_reload[str(tmp_epoch)] = 1
            print('learning keep: ', tmp_epoch)

        pretrained_model = temp_path + 'save_' + str(tmp_epoch) + '_' + str(round(best_ACC, 2)) + '.pth'
        pretrain = torch.load(pretrained_model)
        model_DHCN.load_state_dict(pretrain['state_dict_DHCN'])
        reload_model = False
    # epoch training
    model_DHCN.train()

    # loss calculation
    loss_supervised, loss_self = 0.0, 0.0
    loss = 0.0
    slices = 0
    batch_losses = []
    for TRAIN_IMG, TRAIN_Y, TRAIN_PL in zip(INPUT_DATA[0], INPUT_DATA[1], INPUT_DATA[2]):
        batch_size = 160
        dataloader, first = make_ds(TRAIN_IMG, TRAIN_Y, TRAIN_PL, batch_size)
        #batch loss/train
        for id_batch, (TRAIN_IMG, TRAIN_Y, TRAIN_PL) in enumerate(dataloader):
            TRAIN_IMG, TRAIN_Y, TRAIN_PL = permute_batch(TRAIN_IMG, TRAIN_Y, TRAIN_PL, first)
            scores, _ = model_DHCN(TRAIN_IMG.to(device))
            slices += 1
            loss_self = 0.0
            loss_supervised = 0.0
            for k_Layer in range(DHCN_LAYERS + 1):
                for i_num, (k_scores, k_TRAIN_Y) in enumerate(zip(scores[k_Layer], TRAIN_Y)):
                    k_TRAIN_Y = k_TRAIN_Y.to(device)
                    if len(k_TRAIN_Y[k_TRAIN_Y > 0]) > 0:
                        loss_supervised += loss_ce(k_scores.permute(1, 2, 0)[k_TRAIN_Y > 0], k_TRAIN_Y[k_TRAIN_Y > 0])
                    for id_layer, k_TRAIN_PL in enumerate(TRAIN_PL):
                        k_TRAIN_PL = k_TRAIN_PL.to(device)
                        if (k_TRAIN_PL[i_num].sum(-1) > 1).sum() > 0:
                            i_num
                        else:
                            onehot2label = torch.topk(k_TRAIN_PL[i_num],k=1,dim=-1)[1].squeeze(-1)
                            if len(onehot2label[onehot2label > 0]) > 0:
                                loss_self += (1 / float(id_layer + 1)) * loss_ce(k_scores.permute(1,2,0)[onehot2label > 0], onehot2label[onehot2label > 0])
                    try:
                        if loss_supervised.item() != loss_supervised.item():
                            print('supervised nan')
                            breakpoint()
                        if loss_self.item() != loss_self.item():
                            print('self loss nan')
                            breakpoint()
                    except AttributeError:
                        pass
            # backpropagation
            if USE_HARD_LABELS:
                loss = loss_supervised + loss_self
            else:
                loss = loss_self
            if type(loss) == float:
                pass
            else:
                batch_losses.append(loss.item())
                nn.utils.clip_grad_norm_(model_DHCN.parameters(), 3.0)
                loss.backward()#retain_graph=True)
                optimizer_DHCN.step()
                optimizer_DHCN.zero_grad()
    losses.append(np.mean(batch_losses))
    
    # Test/validation check
    internum = 50
    if epoch < 300:
        internum = 100
    if epoch > 500:
        internum = 10

    #test/validation 
    if epoch % internum == 0:
        model_DHCN.eval()

        p_idx = []
        fusion_prediction = 0.0
        batch_size = 160
        fusion_predictions = []
        for k_data, current_data in enumerate(INPUT_DATA[0]):
            # dataloader, first = make_ds2([current_data], batch_size, shuffle=False)
            if current_data.shape[1] > current_data.shape[2]:
                dataset = TensorDataset(current_data.permute(1,0,2,3))
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                first = True
            else:
                dataset = TensorDataset(current_data.permute(2,1,0,3))
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                first = False
            fusion_prediction_helper = []
            p_idx_for_batches = [[] for i in range(len(dataloader))]
            for id_batch, (current_data) in enumerate(dataloader):
                if first:
                    current_data = current_data[0].permute(1, 3, 0, 2)
                else:
                    current_data = current_data[0].permute(2,3,1,0)
                fusion_prediction_batch = 0.0
                scores, _ = model_DHCN(current_data.to(device))
                if params.ROT == False:
                    for k_score in scores:
                        fusion_prediction_batch += F.softmax(k_score[0].permute(1, 2, 0), dim=-1).cpu().data.numpy()
                else:
                    for k_score in scores:
                        if k_data == 0:
                            fusion_prediction_batch += F.softmax(k_score[0].permute(1, 2, 0), dim=-1).cpu().data.numpy()
                            fusion_prediction_batch += np.rot90(
                                F.softmax(k_score[1].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=2, axes=(0, 1))
                            fusion_prediction_batch += F.softmax(k_score[2].permute(1, 2, 0), dim=-1).cpu().data.numpy()[
                                                ::-1, :, :]
                            fusion_prediction_batch += np.rot90(
                                F.softmax(k_score[3].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=2,
                                axes=(0, 1))[::-1, :, :]

                            p_idx_for_batches[id_batch].append(k_score[0].max(0)[-1].cpu().data.numpy())
                            p_idx_for_batches[id_batch].append(np.rot90(k_score[1].max(0)[-1].cpu().data.numpy(), k=2, axes=(0, 1)))
                            p_idx_for_batches[id_batch].append(k_score[2].max(0)[-1].cpu().data.numpy()[::-1, :])
                            p_idx_for_batches[id_batch].append(
                                np.rot90(k_score[3].max(0)[-1].cpu().data.numpy(), k=2, axes=(0, 1))[::-1, :])
                        if k_data == 1:
                            fusion_prediction_batch += np.rot90(
                                F.softmax(k_score[0].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=-1,
                                axes=(0, 1))
                            fusion_prediction_batch += np.rot90(
                                F.softmax(k_score[1].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=1, axes=(0, 1))
                            fusion_prediction_batch += np.rot90(
                                F.softmax(k_score[2].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=-1,
                                axes=(0, 1))[::-1, :, :]
                            fusion_prediction_batch += np.rot90(
                                F.softmax(k_score[3].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=1,
                                axes=(0, 1))[::-1, :, :]

                            p_idx_for_batches[id_batch].append(np.rot90(k_score[0].max(0)[-1].cpu().data.numpy(), k=-1, axes=(0, 1)))
                            p_idx_for_batches[id_batch].append(np.rot90(k_score[1].max(0)[-1].cpu().data.numpy(), k=1, axes=(0, 1)))
                            p_idx_for_batches[id_batch].append(
                                np.rot90(k_score[2].max(0)[-1].cpu().data.numpy(), k=-1, axes=(0, 1))[::-1, :])
                            p_idx_for_batches[id_batch].append(
                                np.rot90(k_score[3].max(0)[-1].cpu().data.numpy(), k=1, axes=(0, 1))[::-1, :])
                fusion_prediction_helper.append(fusion_prediction_batch)
            if gt.shape[0] > gt.shape[1]:
                p_idx_for_batches = np.concatenate(p_idx_for_batches, axis=1)
                fusion_predictions.append(np.concatenate(fusion_prediction_helper, axis=0))
            else:
                p_idx_for_batches = np.concatenate(p_idx_for_batches, axis=2)
                fusion_predictions.append(np.concatenate(fusion_prediction_helper, axis=1))
            if len(p_idx_for_batches) > 0:
                for i in p_idx_for_batches:
                    p_idx.append(i)
        fusion_predictions = np.array(fusion_predictions)
        fusion_prediction = np.sum(fusion_predictions, axis=0)
        Acc = np.zeros([len(p_idx) + 1])
        for count, k_idx in enumerate(p_idx):
            Acc[count] = metrics(
                k_idx.reshape(img.shape[0], img.shape[1]), test_gt, 
                ignored_labels=IGNORED_LABELS, n_classes=N_CLASSES)['Accuracy']
        Acc[-1] = metrics(
            fusion_prediction.argmax(-1).reshape(img.shape[:2]), test_gt, ignored_labels=IGNORED_LABELS,
            n_classes=N_CLASSES)['Accuracy'] # TODO: This calculates probably wrong thing. Probably fixed TODO: Check
        OA,AA=calprecision(fusion_prediction.argmax(-1).reshape(img.shape[:2]), test_gt,n_classes=N_CLASSES)
        kappa=metrics(fusion_prediction.argmax(-1).reshape(img.shape[:2]), test_gt, ignored_labels=IGNORED_LABELS,
                n_classes=N_CLASSES)['Kappa']

        tmp_count += 1
        accuracies.append(max(Acc))
        if max(Acc) > best_ACC:
            best_ACC = max(Acc)
            temp_name_root = 'save_' + str(epoch) + '_' + str(round(best_ACC, 2))
            states = {'state_dict_DHCN': model_DHCN.state_dict(),
                        'train_gt': train_gt,
                        'test_gt': test_gt, }
            mid_training_saves(temp_path, temp_name_root, states, losses, accuracies)
            tmp_count = 0
            tmp_epoch = epoch
            print('save: ', epoch, str(round(best_ACC, 2)))
            print('save: %d, OA: %f AA: %f Kappa: %f' %(epoch, OA,AA,kappa))
            print(np.round(Acc, 2))

        if tmp_count == max_tmp_count:
            reload_model = True
            tmp_count = 0
    else:
        if len(accuracies) > 0:
            accuracies.append(accuracies[-1])
        else:
            accuracies.append(0)


def train_epoch():
    return None

def train_batch():
    return None


        

def main(params):
    RUNS, MX_ITER, SAMPLE_PERCENTAGE, DATASET, DHCN_LAYERS, CONV_SIZE, \
        H_MIRROR, USE_HARD_LABELS, LR, device, file_and_folder_name_common_part, \
            temp_path, latest_path, best_path, run_numbers = presets(params)
    if RUNS == 0:
        return
    
    for run in run_numbers:

        start_time = time.time()
        accuracies = []
        losses = []


        datasets_root = '/mnt/data/leevi/'
        img, gt, LABEL_VALUES, IGNORED_LABELS, ALL_LABELS, X, Y, \
            N_CLASSES, INPUT_SIZE, train_gt, test_gt, pseudo_labels3 = load_datasets(DATASET, datasets_root, SAMPLE_PERCENTAGE)
        
        trainnum = np.sum(train_gt > 0)
        print("trainnum:%d" % (trainnum))
        INPUT_DATA = pre_data(img, train_gt, params, gt,pseudo_labels3) # Should the batch size be in pre_data? 
        model_DHCN = DHCN(input_size=INPUT_SIZE, embed_size=INPUT_SIZE, densenet_layer=DHCN_LAYERS,
                          output_size=N_CLASSES, conv_size=CONV_SIZE, batch_norm=False).to(device)
        optimizer_DHCN = torch.optim.Adam(model_DHCN.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=1e-4)
        model_DHCN = nn.DataParallel(model_DHCN)
        loss_ce = nn.CrossEntropyLoss().to(device)

        best_ACC, tmp_epoch, tmp_count, tmp_rate, recode_reload, reload_model = 0.0, 0, 0, LR, {}, False
        max_tmp_count = 300

        for epoch in range(MX_ITER):
            if epoch % 100 == 0:
                print("epoch: %d" % (epoch))
            current_time = time.time()
            if current_time - start_time > 3600:
                
                print(f'Training end due to current_time={current_time-start_time}')
                end_of_training_time(
                    model_DHCN, train_gt, test_gt, 
                    file_and_folder_name_common_part, best_ACC, tmp_epoch,
                    temp_path, best_path, latest_path, losses, accuracies, run)
                break

            if reload_model == True:

                if str(tmp_epoch) in recode_reload:

                    recode_reload[str(tmp_epoch)] += 1
                    tmp_rate = tmp_rate * 0.1
                    if tmp_rate < 1e-6:

                        print(f'Training end due to tmp_rate={tmp_rate}')
                        end_of_training_time(
                            model_DHCN, train_gt, test_gt, 
                            file_and_folder_name_common_part, best_ACC, tmp_epoch,
                            temp_path, best_path, latest_path, losses, accuracies, run)
                        break

                    print('learning decay: ', str(tmp_epoch), tmp_rate)
                    for param_group in optimizer_DHCN.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.1

                else:
                    recode_reload[str(tmp_epoch)] = 1
                    print('learning keep: ', tmp_epoch)

                pretrained_model = temp_path + 'save_' + str(tmp_epoch) + '_' + str(round(best_ACC, 2)) + '.pth'
                pretrain = torch.load(pretrained_model)
                model_DHCN.load_state_dict(pretrain['state_dict_DHCN'])
                reload_model = False
            model_DHCN.train()

            loss_supervised, loss_self = 0.0, 0.0
            loss = 0.0
            slices = 0
            batch_losses = []
            for TRAIN_IMG, TRAIN_Y, TRAIN_PL in zip(INPUT_DATA[0], INPUT_DATA[1], INPUT_DATA[2]):
                batch_size = 160
                dataloader, first = make_ds(TRAIN_IMG, TRAIN_Y, TRAIN_PL, batch_size)
                for id_batch, (TRAIN_IMG, TRAIN_Y, TRAIN_PL) in enumerate(dataloader):
                    TRAIN_IMG, TRAIN_Y, TRAIN_PL = permute_batch(TRAIN_IMG, TRAIN_Y, TRAIN_PL, first)
                    scores, _ = model_DHCN(TRAIN_IMG.to(device))
                    slices += 1
                    loss_self = 0.0
                    loss_supervised = 0.0
                    for k_Layer in range(DHCN_LAYERS + 1):
                        for i_num, (k_scores, k_TRAIN_Y) in enumerate(zip(scores[k_Layer], TRAIN_Y)):
                            k_TRAIN_Y = k_TRAIN_Y.to(device)
                            if len(k_TRAIN_Y[k_TRAIN_Y > 0]) > 0:
                                loss_supervised += loss_ce(k_scores.permute(1, 2, 0)[k_TRAIN_Y > 0], k_TRAIN_Y[k_TRAIN_Y > 0])
                            for id_layer, k_TRAIN_PL in enumerate(TRAIN_PL):
                                k_TRAIN_PL = k_TRAIN_PL.to(device)
                                if (k_TRAIN_PL[i_num].sum(-1) > 1).sum() > 0:
                                    i_num
                                else:
                                    onehot2label = torch.topk(k_TRAIN_PL[i_num],k=1,dim=-1)[1].squeeze(-1)
                                    if len(onehot2label[onehot2label > 0]) > 0:
                                        loss_self += (1 / float(id_layer + 1)) * loss_ce(k_scores.permute(1,2,0)[onehot2label > 0], onehot2label[onehot2label > 0])
                            try:
                                if loss_supervised.item() != loss_supervised.item():
                                    print('supervised nan')
                                    breakpoint()
                                if loss_self.item() != loss_self.item():
                                    print('self loss nan')
                                    breakpoint()
                            except AttributeError:
                                pass
                    if USE_HARD_LABELS:
                        loss = loss_supervised + loss_self
                    else:
                        loss = loss_self
                    if type(loss) == float:
                        pass
                    else:
                        batch_losses.append(loss.item())
                        nn.utils.clip_grad_norm_(model_DHCN.parameters(), 3.0)
                        loss.backward()#retain_graph=True)
                        optimizer_DHCN.step()
                        optimizer_DHCN.zero_grad()
            losses.append(np.mean(batch_losses))
            
            internum = 50
            if epoch < 300:
                internum = 100
            if epoch > 500:
                internum = 10

            if epoch % internum == 0:
                model_DHCN.eval()

                p_idx = []
                fusion_prediction = 0.0
                batch_size = 160
                fusion_predictions = []
                for k_data, current_data in enumerate(INPUT_DATA[0]):
                    # dataloader, first = make_ds2([current_data], batch_size, shuffle=False)
                    if current_data.shape[1] > current_data.shape[2]:
                        dataset = TensorDataset(current_data.permute(1,0,2,3))
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                        first = True
                    else:
                        dataset = TensorDataset(current_data.permute(2,1,0,3))
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                        first = False
                    fusion_prediction_helper = []
                    p_idx_for_batches = [[] for i in range(len(dataloader))]
                    for id_batch, (current_data) in enumerate(dataloader):
                        if first:
                            current_data = current_data[0].permute(1, 3, 0, 2)
                        else:
                            current_data = current_data[0].permute(2,3,1,0)
                        fusion_prediction_batch = 0.0
                        scores, _ = model_DHCN(current_data.to(device))
                        if params.ROT == False:
                            for k_score in scores:
                                fusion_prediction_batch += F.softmax(k_score[0].permute(1, 2, 0), dim=-1).cpu().data.numpy()
                        else:
                            for k_score in scores:
                                if k_data == 0:
                                    fusion_prediction_batch += F.softmax(k_score[0].permute(1, 2, 0), dim=-1).cpu().data.numpy()
                                    fusion_prediction_batch += np.rot90(
                                        F.softmax(k_score[1].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=2, axes=(0, 1))
                                    fusion_prediction_batch += F.softmax(k_score[2].permute(1, 2, 0), dim=-1).cpu().data.numpy()[
                                                        ::-1, :, :]
                                    fusion_prediction_batch += np.rot90(
                                        F.softmax(k_score[3].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=2,
                                        axes=(0, 1))[::-1, :, :]

                                    p_idx_for_batches[id_batch].append(k_score[0].max(0)[-1].cpu().data.numpy())
                                    p_idx_for_batches[id_batch].append(np.rot90(k_score[1].max(0)[-1].cpu().data.numpy(), k=2, axes=(0, 1)))
                                    p_idx_for_batches[id_batch].append(k_score[2].max(0)[-1].cpu().data.numpy()[::-1, :])
                                    p_idx_for_batches[id_batch].append(
                                        np.rot90(k_score[3].max(0)[-1].cpu().data.numpy(), k=2, axes=(0, 1))[::-1, :])
                                if k_data == 1:
                                    fusion_prediction_batch += np.rot90(
                                        F.softmax(k_score[0].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=-1,
                                        axes=(0, 1))
                                    fusion_prediction_batch += np.rot90(
                                        F.softmax(k_score[1].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=1, axes=(0, 1))
                                    fusion_prediction_batch += np.rot90(
                                        F.softmax(k_score[2].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=-1,
                                        axes=(0, 1))[::-1, :, :]
                                    fusion_prediction_batch += np.rot90(
                                        F.softmax(k_score[3].permute(1, 2, 0), dim=-1).cpu().data.numpy(), k=1,
                                        axes=(0, 1))[::-1, :, :]

                                    p_idx_for_batches[id_batch].append(np.rot90(k_score[0].max(0)[-1].cpu().data.numpy(), k=-1, axes=(0, 1)))
                                    p_idx_for_batches[id_batch].append(np.rot90(k_score[1].max(0)[-1].cpu().data.numpy(), k=1, axes=(0, 1)))
                                    p_idx_for_batches[id_batch].append(
                                        np.rot90(k_score[2].max(0)[-1].cpu().data.numpy(), k=-1, axes=(0, 1))[::-1, :])
                                    p_idx_for_batches[id_batch].append(
                                        np.rot90(k_score[3].max(0)[-1].cpu().data.numpy(), k=1, axes=(0, 1))[::-1, :])
                        fusion_prediction_helper.append(fusion_prediction_batch)
                    if gt.shape[0] > gt.shape[1]:
                        p_idx_for_batches = np.concatenate(p_idx_for_batches, axis=1)
                        fusion_predictions.append(np.concatenate(fusion_prediction_helper, axis=0))
                    else:
                        p_idx_for_batches = np.concatenate(p_idx_for_batches, axis=2)
                        fusion_predictions.append(np.concatenate(fusion_prediction_helper, axis=1))
                    if len(p_idx_for_batches) > 0:
                        for i in p_idx_for_batches:
                            p_idx.append(i)
                fusion_predictions = np.array(fusion_predictions)
                fusion_prediction = np.sum(fusion_predictions, axis=0)
                Acc = np.zeros([len(p_idx) + 1])
                for count, k_idx in enumerate(p_idx):
                    Acc[count] = metrics(
                        k_idx.reshape(img.shape[0], img.shape[1]), test_gt, 
                        ignored_labels=IGNORED_LABELS, n_classes=N_CLASSES)['Accuracy']
                Acc[-1] = metrics(
                    fusion_prediction.argmax(-1).reshape(img.shape[:2]), test_gt, ignored_labels=IGNORED_LABELS,
                    n_classes=N_CLASSES)['Accuracy'] # TODO: This calculates probably wrong thing. Probably fixed TODO: Check
                OA,AA=calprecision(fusion_prediction.argmax(-1).reshape(img.shape[:2]), test_gt,n_classes=N_CLASSES)
                kappa=metrics(fusion_prediction.argmax(-1).reshape(img.shape[:2]), test_gt, ignored_labels=IGNORED_LABELS,
                        n_classes=N_CLASSES)['Kappa']

                tmp_count += 1
                accuracies.append(max(Acc))
                if max(Acc) > best_ACC:
                    best_ACC = max(Acc)
                    temp_name_root = 'save_' + str(epoch) + '_' + str(round(best_ACC, 2))
                    states = {'state_dict_DHCN': model_DHCN.state_dict(),
                              'train_gt': train_gt,
                              'test_gt': test_gt, }
                    mid_training_saves(temp_path, temp_name_root, states, losses, accuracies)
                    tmp_count = 0
                    tmp_epoch = epoch
                    print('save: ', epoch, str(round(best_ACC, 2)))
                    print('save: %d, OA: %f AA: %f Kappa: %f' %(epoch, OA,AA,kappa))
                    print(np.round(Acc, 2))

                if tmp_count == max_tmp_count:
                    reload_model = True
                    tmp_count = 0
            else:
                if len(accuracies) > 0:
                    accuracies.append(accuracies[-1])
                else:
                    accuracies.append(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Low shot benchmark')
    parser.add_argument('--DHCN_LAYERS', default=1, type=int)
    parser.add_argument('--SAMPLE_PERCENTAGE', default=5, type=int)
    parser.add_argument('--DATASET', default="IndianPines", type=str)  # KSC, PaviaU, IndianPines, Botswana,    !!PaviaC
    parser.add_argument('--CONV_SIZE', default=3, type=int)  # 3,5,7
    parser.add_argument('--ROT', default=True, type=bool)  # False
    parser.add_argument('--MIRROR', default=True, type=bool)  # False
    parser.add_argument('--H_MIRROR', default='full', type=str)  # half, full
    parser.add_argument('--GPU', default='0,1,2,3', type=str)  # 0,1,2,3
    parser.add_argument('--ROT_N', default=1, type=int)  # False
    parser.add_argument('--MIRROR_N', default=1, type=int)  # False
    parser.add_argument('--USE_HARD_LABELS_N', default=1,type=int)
    return parser.parse_args()


if __name__ == '__main__':
    params = parse_args()
    params.ROT = True if params.ROT_N == 1 else False
    params.MIRROR = True if params.MIRROR_N == 1 else False
    params.USE_HARD_LABELS = True if params.USE_HARD_LABELS_N == 1 else False
    main(params)
