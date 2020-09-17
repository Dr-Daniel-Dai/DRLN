#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:22:39 2018
move all images in folds including subfolds to a certain fold
@author: dy
"""

import os  
import shutil  
path = './CUB_200_2011'      ####   headset   handband   watch   VR
new_path = './CUB/data'  

isExists=os.path.exists(new_path)
if not isExists:
    os.makedirs(new_path)

for root, dirs, files in os.walk(path):  
    if len(dirs) == 0:  
        for i in range(len(files)):  
            print(i)
            if files[i][-3:] == 'jpg':  
                file_path = root+'/'+files[i]  
                new_file_path = new_path+ '/'+ files[i]  
                shutil.move(file_path,new_file_path)
                # shutil.copy(file_path,new_file_path)