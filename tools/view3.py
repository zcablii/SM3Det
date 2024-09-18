import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import os
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import seaborn as sns
import mmrotate
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config',default='/root/lyx/LSKNet/work_dirs/ablation_moe_blk_convnext_t_orcnn_gfl_e8t2_evenblocks/ablation_moe_blk_convnext_t_orcnn_gfl_e8t2_evenblocks_show.py',help='Config file')
    parser.add_argument('--checkpoint',default='/root/lyx/LSKNet/work_dirs/ablation_moe_blk_convnext_t_orcnn_gfl_e8t2_evenblocks/iter_33468.pth',help='Checkpoint file')
    parser.add_argument('--img',default='/root/lyx/LSKNet/show_expert/picture',help='Image file')
    parser.add_argument('--experts_id',default='/root/lyx/LSKNet/show_expert/temp',help='expert_id file')
    parser.add_argument('--outfile',default='/root/lyx/LSKNet/show_expert/statistical',help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args
# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('--config',default='/root/lyx/LSKNet/work_dirs/old/trisource_convnext_t_orcnn_gfl_e4t2_lr0001/trisource_convnext_t_orcnn_gfl_e4t2_lr0001.py',help='Config file')
#     parser.add_argument('--checkpoint',default='/root/lyx/LSKNet/work_dirs/old/trisource_convnext_t_orcnn_gfl_e4t2_lr0001/iter_33468.pth',help='Checkpoint file')
#     parser.add_argument('--img',default='/root/lyx/LSKNet/show_expert/test',help='Image file')
#     parser.add_argument('--experts_id',default='/root/lyx/LSKNet/show_expert/temp',help='expert_id file')
#     parser.add_argument('--outfile',default='/root/lyx/LSKNet/show_expert/expert',help='Path to output file')
#     parser.add_argument('--device', default='cuda:0', help='Device used for inference')
#     args = parser.parse_args()
#     return args
color0=[255,0,255]#粉色

color_background2=[255,255,0]#浅蓝

color_object=[0,255,255]#黄色

color3=[0,255,0]#绿色

color_background1=[245,245,245]#白色

color5=[255,0,0]#蓝色？

color6=[139,61,72]#紫色

color7=[0,0,255]#红色
color_black=[0,0,0]
Color=[]
Color.append([color_black,color0,color_background2,color_object,color5,color_background1,color3,color6,color7])

Color.append([color_black,color0,color_background1,color_background2,color3,color_object,color6,color5,color7])

Color.append([color_black,color_background2,color_object,color0,color5,color_background1,color3,color6,color7])

Color.append([color_black,color_background2,color5,color0,color_object,color_background1,color3,color6,color7])
Color_name=[]
Color_name.append(['Pink','Light blue','Yellow','Blue','White','Green','Purple','Red'])
Color_name.append(['Pink','White','Light blue','Green','Yellow','Purple','Blue','Red'])
Color_name.append(['Light blue','Yellow','Pink','Blue','White','Green','Purple','Red'])
Color_name.append(['Light blue','Blue','Pink','Yellow','White','Green','Purple','Red'])
expert_num=4
ls=[2,8,9,10]
def pic(model,savedir,filename,filedir,temp):
    
    imgname=filename.split('.')
    imgname=imgname[0]
    path=f'{savedir}/{imgname}'
    if not os.path.exists(path):
        os.makedirs(path)
   
    print(filedir)
    result= inference_detector(model, filedir)
    # ls=[2,8,9,10]
    for tt in range(0,len(ls)):
        id=ls[tt]
        experts=torch.load(temp+f'/expert_id_{id}.pt').detach()
        print(experts)
        experts=experts.cpu()
        experts=experts.long()
        for jj in range(1,6):
            plt.subplots(figsize=(25, 20))
            ax = sns.heatmap(experts[jj-1][1:,1:],linewidths=0.3,fmt="d", annot=True,cmap="RdBu_r")
            ax.set_xlabel("First", fontsize=20)
            ax.set_ylabel("Second", fontsize=20)
            ax.set_xticklabels(Color_name[tt],fontsize=15)
            ax.set_yticklabels(Color_name[tt],fontsize=15)
            plt.savefig(f'{path}/expert_id_{id}_0.{jj}_HeatMap.jpg',dpi=300,bbox_inches='tight')

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