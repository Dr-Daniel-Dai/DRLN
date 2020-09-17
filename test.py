#coding=utf-8
import os
import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from transforms import transforms
from models.LoadModel import MainModel
from utils.dataset_DCL_index import collate_fn4train, collate_fn4test, collate_fn4val, dataset
from config import LoadConfig, load_data_transformers
from utils.test_tool import set_text, save_multi_img, cls_base_acc

import pdb

from sklearn.metrics import label_ranking_average_precision_score, average_precision_score
from utils.eval_performance_dy import evaluate_test

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='PID', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='densenet121', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=10, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=0, type=int)
    parser.add_argument('--ver', dest='version',
                        default='test', type=str)
    parser.add_argument('--save_dir', dest='resume',
                        default=None, type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--ss', dest='save_suffix',
                        default='pth', type=str)
    parser.add_argument('--acc_report', dest='acc_report',
                        default = True)    ###   action='store_true'
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    parser.add_argument('--submit', default=True, type=str)
    args = parser.parse_args()
    return args


def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ','')) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    m_dir = os.path.join(load_dir, choosed, choosed_w)
    print (choosed_w)
    return m_dir.replace('\\','/')


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.submit:
        args.version = 'train'
        if args.save_suffix == '':
            raise Exception('**** miss --ss save suffix is needed. ')

    Config = LoadConfig(args, args.version)
    
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)
    
    data_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        common_aug = transformers["common_aug"],\
                        swap = transformers["swap"],\
                        totensor = transformers["val_totensor"],\
                        train = False)  

    dataloader = torch.utils.data.DataLoader(data_set,\
                                             batch_size=args.batch_size,\
                                             shuffle=False,\
                                             num_workers=args.num_workers,\
                                             collate_fn=collate_fn4val)

    setattr(dataloader, 'total_item_len', len(data_set))

    cudnn.benchmark = True

    model = MainModel(Config)
    model_dict=model.state_dict()
    
    resume = auto_load_resume(Config.save_dir)
    
    pretrained_dict=torch.load(resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model = nn.DataParallel(model)

    model.train(False)
    
    ave_test_accu_final = 0
    
    
    eval_t = locals()
    test_file = open("./result_log/test.log", "a+")
    
    
    with torch.no_grad():
        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0
        val_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        count_bar = tqdm(total=dataloader.__len__())
        
        for i in range(50,51):
            y_pred ,Y_test, y_pred_dy =  [], [], []
            sum_fbeta = 0
            best_fbeta = 0
            
            for batch_cnt_val, data_val in enumerate(dataloader):
                
                count_bar.update(1)
                inputs, labels, labels_swap, swap_law, law_index, img_names = data_val   
            
                # labels_npy = np.array(labels)
                # labels_tensor = Variable(torch.FloatTensor(np.array(labels)).cuda())
                
                labels_tensor_ = Variable(torch.FloatTensor(np.array(labels)).cuda())
                
                idx_unswap = torch.tensor([0,2,4,6,8],dtype=torch.long).cuda()
                labels_tensor = torch.index_select(labels_tensor_,dim = 0, index= idx_unswap)
    
                labels_npy = np.array(labels_tensor.cpu())
                labels_ = labels_npy.astype(np.uint8)
                
                inputs = Variable(inputs.cuda())
           
                outputs = model(inputs, law_index)
                # print (outputs[0].shape)

                outputs_pred = outputs[0]
                
                # cal_sigmoid = nn.Sigmoid()
                # outputs_pred_s = cal_sigmoid(outputs_pred)   
                ########  MAP is label ranking, do not need normilization
                # predict_multensor = torch.ge(outputs_pred, 0.5)     ###   大于0.5的置为一，其他置为0，类似于阈值化操作
                predict_mul_ = outputs_pred.cpu().numpy()
                
                temp_fbeta = label_ranking_average_precision_score(labels_, predict_mul_)
                
                temp_threshhold = 0.5
                
                predict_multensor = torch.ge(outputs_pred, temp_threshhold)     ###   大于0.5的置为一，其他置为0，类似于阈值化操作
                predict_mul = predict_multensor.cpu().numpy()
                
                sum_fbeta = sum_fbeta + temp_fbeta
                ave_num = batch_cnt_val + 1
                
                y_pred.extend(predict_mul[  : ] )
                y_pred_dy.extend(predict_mul_[  : ] )
                Y_test.extend(labels_[  : ] )
            
                if best_fbeta < temp_fbeta:
                    best_fbeta = temp_fbeta
                    
            ave_acc = sum_fbeta/ave_num
            best_test_accu_temp=(" bestp-batch:%6.4f ,  %g "% (best_fbeta, temp_threshhold) )
            print(best_test_accu_temp)
                    
            if ave_acc >= ave_test_accu_final :
                ave_test_accu_final = ave_acc
                temp_threshhold_final = temp_threshhold
                
                y_pred_ = np.array(y_pred)
                y_pred_dy_ = np.array(y_pred_dy)
                Y_test_ = np.array(Y_test)
                
                np.save('result_log/Y_pred-DRLN.npy', y_pred_dy_)
                np.save('result_log/Y_test-DRLN.npy', Y_test_)
        
                eval_t['metrics_'+ str(temp_threshhold)] = evaluate_test(predictions=y_pred_, labels=Y_test_)
                
        ave_best=("---aveAccu_final: %10.5g , ave_threshold_final: %10.5g---\n "% (ave_test_accu_final, temp_threshhold_final) )
        print(ave_best) 
            
        metrics = eval_t['metrics_'+ str(temp_threshhold_final)]
        output = "=> Test : Coverage = {}\n Average Precision = {}\n Micro Precision = {}\n Micro Recall = {}\n Micro F Score = {}\n".format(metrics['coverage'], ave_test_accu_final, metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'])
        output += "\n=> Test : Macro Precision = {}\n Macro Recall = {}\n Macro F Score = {}\n ranking_loss = {}\n hamming_loss = {}\n\n".format(metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'], metrics['ranking_loss'], metrics['hamming_loss'])
        # output += "\n=> Test : ma-False_positive_rate(FPR) = {}, mi-False_positive_rate(FPR) = {}\n".format(metrics['ma-FPR'],metrics['mi-FPR'])
        print(output)
        test_file.write(output)
        test_file.close()

    count_bar.close()

            

            

