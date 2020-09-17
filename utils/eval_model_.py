#coding=utf8
from __future__ import print_function, division
import os,time,datetime
import numpy as np
import datetime
from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.utils import LossRecord

import pdb

from sklearn.metrics import label_ranking_average_precision_score, average_precision_score


def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def get_sigmoid_ce(predictions,target):
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    predictions=m(predictions)
    # target = torch.tensor(target, dtype=torch.float)
    
    output = loss(predictions, target.cuda())
    
    return output

def eval_turn(Config, model, data_loader, val_version, epoch_num, log_file):

    model.train(False)

    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0
    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()
    get_l1_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()

    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    num_cls = data_loader.num_cls

    val_loss_recorder = LossRecord(val_batch_size)
    val_celoss_recorder = LossRecord(val_batch_size)
    print('evaluating %s ...'%val_version, flush=True)
    
    sum_fbeta = 0
    
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            
            # inputs = Variable(data_val[0].cuda())
            # labels = Variable(torch.LongTensor(np.array(data_val[1])).long().cuda())
            
            inputs, labels, labels_swap, swap_law, img_names = data_val
            labels_npy = np.array(labels)
            
            labels_tensor = Variable(torch.FloatTensor(labels_npy).cuda())
            
            labels_ = labels_npy.astype(np.uint8)
            
            inputs = Variable(inputs.cuda())
                        
            outputs = model(inputs)
            loss = 0

            # ce_loss = get_ce_loss(outputs[0], labels).item()
            ce_loss = get_sigmoid_ce(outputs[0], labels_tensor).item()
            loss += ce_loss

            val_loss_recorder.update(loss)
            val_celoss_recorder.update(ce_loss)

            if Config.use_dcl and Config.cls_2xmul:
                outputs_pred = outputs[0] + outputs[1][:,0:num_cls] + outputs[1][:,num_cls:2*num_cls]
            else:
                outputs_pred = outputs[0]
            ########  MAP is label ranking, do not need normilization
            # predict_multensor = torch.ge(outputs_pred, 0.5)     ###   大于0.5的置为一，其他置为0，类似于阈值化操作
            predict_mul = outputs_pred.cpu().numpy()
            
            temp_fbeta = label_ranking_average_precision_score(labels_, predict_mul)
            #################################################################  dy modify    Micro precision
            # cor_sum = 0
            # num_sum =0
            
            # for j in range(10): 

            #     query_col = labels_[j,:]
            #     label_col = predict_mul[j,:]
                
            #     index = np.where(label_col > 0.5)
            #     index_ = index[0]
            #     number_=index_.size
                
            #     query_binary = query_col[index]
            #     query_label = label_col[index]
                
            #     batch_corrects1 = np.count_nonzero(query_binary == query_label)
                
            #     cor_sum = cor_sum + batch_corrects1
            #     num_sum = num_sum + number_
                
            # temp_fbeta = cor_sum/num_sum
            ##################################################################
            
            sum_fbeta = sum_fbeta + temp_fbeta
            ave_num = batch_cnt_val + 1
            
            
            # top3_val, top3_pos = torch.topk(outputs_pred, 3)

            # print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, loss), flush=True)

        #     batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
        #     val_corrects1 += batch_corrects1
            
        #     batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
        #     val_corrects2 += (batch_corrects2 + batch_corrects1)
        #     batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
        #     val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

        # val_acc1 = val_corrects1 / item_count
        # val_acc2 = val_corrects2 / item_count
        # val_acc3 = val_corrects3 / item_count

        # log_file.write(val_version  + '\t' +str(val_loss_recorder.get_val())+'\t' + str(val_celoss_recorder.get_val()) + '\t' + str(val_acc1) + '\t' + str(val_acc3) + '\n')

        # t1 = time.time()
        # since = t1-t0
        # print('--'*30, flush=True)
        # print('% 3d %s %s %s-loss: %.4f ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, val_acc1,val_version, val_acc2, val_version, val_acc3, since), flush=True)
        # print('--' * 30, flush=True)

    # return val_acc1, val_acc2, val_acc3
    
        ave_acc = sum_fbeta/ave_num
        log_file.write(val_version  + '\t' +str(val_loss_recorder.get_val())+'\t' + str(val_celoss_recorder.get_val()) + '\t' + str(ave_acc) + '\n')

        t1 = time.time()
        since = t1-t0
        print('--'*30, flush=True)
        print('% 3d %s %s %s-loss: %.4f ||%s-ave@acc: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, ave_acc, since), flush=True)
        print('--' * 30, flush=True)
        
    return ave_acc

