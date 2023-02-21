#!/bin/bash

VIDEO_PATH = /data/ppk/ufs/link/video_new  #视频路径
KEYPOINT_PATH = /home/hehaowen/data/ufs_2d_detect #2D 关键点路径

for file in ${KEYPOINT_PATH}/* 
do 
    exec 
    echo "${file%*.npz}" ;

done
