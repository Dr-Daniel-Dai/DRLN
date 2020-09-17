#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:05:31 2018

@author: dy
"""
import numpy as np
import fileinput




dataname = ['glasses','handband','interphone','mobile','MP4','radio' ,'watch','ru_earphone','sai_earphone','tou_earphone','blue_audio','hair_dryer','humidifier','mouse'] 

ma = 14   ###  14 categories + 4 labels
band = 0
number = 1600

pl = np.zeros([number*ma,ma])

step = 2

if step ==1:
    output_trY = open("dy_train_label.txt", "w")
    
    for name in dataname:
        
        print (band)
        
        for lines in range(number):
            
            pl[number*band + lines, band] = 1
            
            id_band = lines
            output_trY.write(name + '_%d'%id_band +'_jpg ' +str(pl[number*band + lines,:])+'\n')
            
        band = band + 1
        
    output_trY.close()

if step ==2:

    for line in fileinput.input('dy_train_label.txt', backup='.txt',inplace = True):
        print(line.rstrip().replace('jpg', '.jpg'))     ###   [  .  ]


    