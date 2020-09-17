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
from utils.eval_performance_dy import evaluate_test

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

# def get_sigmoid_ce(predictions,target):
#     m = nn.Sigmoid()
#     loss = nn.BCELoss()
#     predictions=m(predictions)
#     # target = torch.tensor(target, dtype=torch.float)
#     output = loss(predictions, target.cuda())
    
#     return output

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
    get_ce_sig_loss = nn.BCELoss()
    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    num_cls = data_loader.num_cls

    val_loss_recorder = LossRecord(val_batch_size)
    val_celoss_recorder = LossRecord(val_batch_size)
    print('evaluating %s ...'%val_version, flush=True)
    
    eval_t = locals()
    sum_fbeta = 0
    y_pred ,Y_test =  [], []
    sum_fbeta = 0
    best_fbeta = 0
    ave_test_accu_final = 0
    test_file = open("./result_log/val.log", "a+")
    
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            # inputs = Variable(data_val[0].cuda())
            # labels = Variable(torch.LongTensor(np.array(data_val[1])).long().cuda())
            
            inputs, labels, labels_swap, swap_law, img_names = data_val
            labels_npy = np.array(labels)
            
            labels_tensor = Variable(torch.FloatTensor(np.array(labels)).cuda())
            
            labels_ = labels_npy.astype(np.uint8)
            
            inputs = Variable(inputs.cuda())
                        
            outputs = model(inputs)
            loss = 0

            # ce_loss = get_ce_loss(outputs[0], labels).item()
            ce_loss = get_ce_sig_loss(outputs[0], labels_tensor).item()
            loss += ce_loss

            val_loss_recorder.update(loss)
            val_celoss_recorder.update(ce_loss)

            if Config.use_dcl and Config.cls_2xmul:
                outputs_pred = outputs[0] + outputs[1][:,0:num_cls] + outputs[1][:,num_cls:2*num_cls]
            else:
                outputs_pred = outputs[0]
                
            # cal_sigmoid = nn.Sigmoid()
            # outputs_pred_s = cal_sigmoid(outputs_pred) 
            ########  MAP is label ranking, do not need normilization
            # predict_multensor = torch.ge(outputs_pred, 0.5)     ###   大于0.5的置为一，其他置为0，类似于阈值化操作
            predict_mul_ = outputs_pred.cpu().numpy()
                        
            temp_fbeta = label_ranking_average_precision_score(labels_, predict_mul_)
            
            predict_multensor = torch.ge(outputs_pred, 0.5)     ###   大于0.5的置为一，其他置为0，类似于阈值化操作
            predict_mul = predict_multensor.cpu().numpy()
            
            sum_fbeta = sum_fbeta + temp_fbeta
            ave_num = batch_cnt_val + 1
            
            y_pred.extend(predict_mul[  : ] )
            Y_test.extend(labels_[  : ] )
        
        ave_acc = sum_fbeta/ave_num 
        
        y_pred_ = np.array(y_pred)
        Y_test_ = np.array(Y_test)
        
        log_file.write(val_version  + '\t' +str(val_loss_recorder.get_val())+'\t' + str(val_celoss_recorder.get_val()) + '\t' + str(ave_acc) + '\n')

        t1 = time.time()
        since = t1-t0
        print('--'*30, flush=True)
        print('% 3d %s %s %s-loss: %.4f ||%s-ave@acc: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, ave_acc, since), flush=True)
        print('--' * 30, flush=True)
        
        eval_t['metrics_'+ str(0.5)] = evaluate_test(predictions=y_pred_, labels=Y_test_)
        
        metrics = eval_t['metrics_'+ str(0.5)]
        
        output = "=> Test : Coverage = {}\n Average Precision = {}\n Micro Precision = {}\n Micro Recall = {}\n Micro F Score = {}\n".format(metrics['coverage'], ave_acc, metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'])
        output += "=> Test : Macro Precision = {}\n Macro Recall = {}\n Macro F Score = {}\n ranking_loss = {}\n hamming_loss = {}\n\n".format(metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'], metrics['ranking_loss'], metrics['hamming_loss'])
        # output += "\n=> Test : ma-False_positive_rate(FPR) = {}, mi-False_positive_rate(FPR) = {}\n".format(metrics['ma-FPR'],metrics['mi-FPR'])
        print(output)
        test_file.write('epoch:%d\n'%epoch_num)
        test_file.write(output)
        test_file.close()
       
        
        
    return ave_acc

