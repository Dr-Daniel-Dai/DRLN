#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:05:31 2018

@author: dy
"""
import numpy as np
import fileinput
from PIL import Image
import random

def swap(img, crop):
    def crop_image(image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    widthcut, highcut = img.size
    # img = img.crop((10, 10, widthcut-10, highcut-10))
    images = crop_image(img, crop)
    index_x = []
    index_y = []
    idx_dy_x = []
    
    
    pro = 5
    if pro >= 5:          
        tmpx = []
        tmpy = []
        count_x = 0
        count_y = 0
        k = 7
        RAN = 2
        for i in range(crop[1] * crop[0]):
            tmpx.append(images[i])
            
            count_x += 1
            index_x.append(count_x-1)
            
            if len(tmpx) >= k:
                
                # tmp = tmpx[count_x - RAN:count_x]
                tmp = tmpx[0:7]
                # print (count_x - RAN)
                
                shuffling = list(zip(index_x, tmp))
                # random.shuffle(tmp)
                random.shuffle(shuffling)
                index_x, tmp = zip(*shuffling) 
                index_x =list(index_x)
                print(index_x)
                # tmpx[count_x - RAN:count_x] = tmp
                tmpx[0:7] = tmp
                
            if count_x == crop[0]:
                tmpy.append(tmpx)
                count_x = 0
                count_y += 1
                index_y.append(count_y-1)
                
                tmpx = []
                idx_dy_x.extend(index_x)
                index_x =[]
                
            if len(tmpy) >= k:
                # tmp2 = tmpy[count_y - RAN:count_y]
                tmp2 = tmpy[0:7]
                
                # random.shuffle(tmp2)
                shuffling = list(zip(index_y, tmp2))
                # random.shuffle(tmp)
                random.shuffle(shuffling)
                index_y, tmp2 = zip(*shuffling) 
                index_y =list(index_y)
                
                # tmpy[count_y - RAN:count_y] = tmp2
                tmpy[0:7] = tmp2
                
        random_im = []
        for line in tmpy:
            random_im.extend(line)
        
        # random.shuffle(images)
        width, high = img.size
        iw = int(width / crop[0])
        ih = int(high / crop[1])
        toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
        x = 0
        y = 0
        nameno = 0
        for i in random_im:
            i = i.resize((iw, ih), Image.ANTIALIAS)
            toImage.paste(i, (x * iw, y * ih))
            i.save('./PID/destruct/%d.jpg'%nameno)
            nameno += 1
            x += 1
            if x == crop[0]:
                x = 0
                y += 1
    else:
        toImage = img
    toImage = toImage.resize((widthcut, highcut))
    # return toImage
    return toImage, idx_dy_x, index_y


def pil_loader(imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

if __name__ == '__main__':
    # img_path = './PID/data/glasses_0.jpg'
    img_path = './PID/OK.jpg'
    img = pil_loader(img_path)
    swap_size=[7,7]
    swap_law2_index = []
    swap_dy_index = []
    toImage, idx_dy_x, idx_dy_y = swap(img, swap_size)
    
    toImage.save('./PID/destruct.jpg')
    
    for idy in range(len(idx_dy_y)):
        for idx in range(len(idx_dy_y)):
            
            index_ = idx_dy_y[idy]*7 + idx_dy_x[idx_dy_y[idy]*7+idx]
            
            swap_dy_index.append(index_)
            
    for h in range(49):
        index = swap_dy_index.index(h)
        swap_law2_index.append(index)
