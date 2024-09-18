import torch
import numpy as np
import cv2
import copy
import os
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config',default='/root/lyx/LSKNet/work_dirs/ablation_moe_blk_convnext_t_orcnn_gfl_e8t2_evenblocks/ablation_moe_blk_convnext_t_orcnn_gfl_e8t2_evenblocks_show.py',help='Config file')
    parser.add_argument('--checkpoint',default='/root/lyx/LSKNet/work_dirs/ablation_moe_blk_convnext_t_orcnn_gfl_e8t2_evenblocks/iter_33468.pth',help='Checkpoint file')
    parser.add_argument('--img',default='/root/lyx/LSKNet/show_expert/test',help='Image file')
    parser.add_argument('--experts_id',default='/root/lyx/LSKNet/show_expert/temp',help='expert_id file')
    parser.add_argument('--outfile',default='/root/lyx/LSKNet/show_expert/expert',help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args
color0=[255,0,255]#粉色

color_background2=[255,255,0]#浅蓝

color_object=[0,255,255]#黄色

color3=[0,255,0]#绿色

color_background1=[245,245,245]#白色

color5=[255,0,0]#蓝色？

color6=[139,61,72]#紫色

color7=[0,0,255]#红色

Color=[]
Color.append([color0,color_background2,color_object,color5,color_background1,color3,color6,color7])

Color.append([color0,color_background1,color_background2,color3,color_object,color6,color5,color7])

Color.append([color_background2,color_object,color0,color5,color_background1,color3,color6,color7])

Color.append([color_background2,color5,color0,color_object,color_background1,color3,color6,color7])
def pic(model,savedir,filename,filedir,temp):
    
    imgname=filename.split('.')
    imgname=imgname[0]
    path1=f'{savedir}/mix/{imgname}'
    path2=f'{savedir}/pure/{imgname}'
    if not os.path.exists(path1):
        os.makedirs(path1+'/0.3')
        os.makedirs(path1+'/0.7')
    # else:
    #     return
    if not os.path.exists(path2):
        os.makedirs(path2)
    print(filedir)
    result= inference_detector(model, filedir)
    img= cv2.imread(filedir)
    cv2.imwrite(f'{path1}/0.3/{imgname}.jpg', img)
    cv2.imwrite(f'{path1}/0.7/{imgname}.jpg', img)
    cv2.imwrite(f'{path2}/{imgname}.jpg', img)
    ls=[2,8,9,10]
    for tt in range(0,4):
        id=ls[tt]
        experts_id=torch.load(temp+f'/expert_id_{id}.pt')
        img= cv2.imread(filedir)
        # print(experts_id.shape)
        experts_id=experts_id.reshape(experts_id.shape[-2],experts_id.shape[-1])
        print(experts_id)
        print(img.shape)
        thickness = 1 
        lineType = 1
        point_color= (1, 1, 1)
        img=cv2.resize(img,dsize=((img.shape[1]//experts_id.shape[1])*experts_id.shape[1],(img.shape[0]//experts_id.shape[0])*experts_id.shape[0]),interpolation=cv2.INTER_LINEAR)
        color = np.zeros((8,img.shape[0]//experts_id.shape[0]-1,img.shape[1]//experts_id.shape[1]-1,3),dtype=np.uint8)
        for t in range(0,8):
            color[t,:,:,0],color[t,:,:,1],color[t,:,:,2]=Color[tt][t]
        step0,step1=int(img.shape[0]//experts_id.shape[0]),int(img.shape[1]//experts_id.shape[1])
        for i in range(0,experts_id.shape[0]):
            ptStart=(0,i*step0)
            ptEnd=(img.shape[1],i*step0)
            cv2.line(img, ptStart, ptEnd, point_color, thickness)
        for i in range(0,experts_id.shape[1]):
            ptStart=(i*step1,0)
            ptEnd=(i*step1,img.shape[0])
            cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
        img1=copy.deepcopy(img)
        img2=copy.deepcopy(img)
        img3=copy.deepcopy(img)
        print(step0,step1)
        if step0==1 or step1==1:
            continue
        for i in range(0,experts_id.shape[0]):
            for j in range(0,experts_id.shape[1]):
                img_add1=cv2.addWeighted(img[i*step0+1:(i+1)*step0,j*step1+1:(j+1)*step1],0.3,color[int(experts_id[i][j])],0.7,0)
                img1[i*step0+1:(i+1)*step0,j*step1+1:(j+1)*step1]=img_add1
                img_add2=cv2.addWeighted(img[i*step0+1:(i+1)*step0,j*step1+1:(j+1)*step1],0.7,color[int(experts_id[i][j])],0.3,0)
                img2[i*step0+1:(i+1)*step0,j*step1+1:(j+1)*step1]=img_add2
                img_add3=cv2.addWeighted(img[i*step0+1:(i+1)*step0,j*step1+1:(j+1)*step1],0,color[int(experts_id[i][j])],1,0)
                img3[i*step0+1:(i+1)*step0,j*step1+1:(j+1)*step1]=img_add3
        #cv2.imshow('image', img)

        cv2.imwrite(f'{path1}/0.7/{imgname}_expert_{id}.jpg', img1)
        cv2.imwrite(f'{path1}/0.3/{imgname}_expert_{id}.jpg', img2)
        cv2.imwrite(f'{path2}/{imgname}_expert_{id}.jpg', img3)
def main(args):

        model = init_detector(args.config, args.checkpoint, device=args.device)
        model.eval()
        for dirpath,dirnames,filenames in os.walk(args.img):
            for filename in filenames:
                savedir=dirpath.split('/')
                savedir=savedir[-1]
                savedir=args.outfile+'/'+savedir
                pic(model,savedir,filename,os.path.join(dirpath,filename),args.experts_id)
        '''
        result= inference_detector(model, args.img)
        imgname=args.img.split('/')
        imgname=imgname[-1]
        imgname=imgname.split('.')
        imgname=imgname[0]
        path=f'{args.outfile}/{imgname}'
        if not os.path.exists(path):
            os.makedirs(path)
        
        '''
if __name__ == '__main__':
    args = parse_args()
    main(args)