import numpy as np
import pdb
import logging
from common.coco_dataset import coco_h36m
'''检测头部正确姿态，正确函数'''
def physical_begin(X):
    kpt=X
    phiscal_center=kpt[2,10]>kpt[2,0] and kpt[2,8]>kpt[2,0] and kpt[2,7]>kpt[2,0] and kpt[2,2]<kpt[2,0] and kpt[2,5]<kpt[2,0] and kpt[2,3]<kpt[2,0] and kpt[2,6]<kpt[2,0]
        #手与膝盖的关系，人体主体与中心点的关系
    #print(phiscal_center)
    phiscal_con_hand=kpt[2,16]>kpt[2,2] and kpt[2,13]>kpt[2,5] 
    #print(phiscal_con_hand)
    phical_con_foot=kpt[2,2]>kpt[2,3] and kpt[2,5]>kpt[2,6]
    
    if phiscal_center and  phiscal_con_hand and  phical_con_foot:
        return True
    else:
        return False
def physical_begin2d(X):
    kpt=X
    # 1,10<1,0 即 左脚踝低于髋关节， 1,8<1,9 即 右脚踝低于髋关节， 1,7<1,0 膝关节低于髋关节，1,2>1,0 ？, 1,5>1,0 ? , 1,3>1,0
    phiscal_center=kpt[1,10]<kpt[1,0] and kpt[1,8]<kpt[1,0] and kpt[1,7]<kpt[1,0] and kpt[1,2]>kpt[1,0] and kpt[1,5]>kpt[1,0] and kpt[1,3]>kpt[1,0] and kpt[1,6]>kpt[1,0]
        #手与膝盖的关系，人体主体与中心点的关系
    #print(phiscal_center)
    phiscal_con_hand=kpt[1,16]<kpt[1,2] and kpt[1,13]<kpt[1,5] 
    #print(phiscal_con_hand)
    phical_con_foot=kpt[1,2]<kpt[1,3] and kpt[1,5]<kpt[1,6]
    
    if phiscal_center and  phiscal_con_hand and  phical_con_foot:
        return True
    else:
        return False
def find_headPose(X2d,X,x,x2d):
    for i in range(X.shape[0]):
        kpt=X[i]
        kpt2d=X2d[i]
        #  physical_begin(kpt) and
        # if i%10==0:
        #     logging.info(f"Headpose(i): physical_begin: {physical_begin(kpt)};find_bone_length: {find_bone_length(x[i])};physical_begin2d: {physical_begin2d(kpt2d)};find_bone_length2d: {find_bone_length(x2d[i])}")    
        if  find_bone_length(x[i]) and physical_begin2d(kpt2d) and find_bone_length(x2d[i]) :  
            return i 
    logging.info(f"Headpose: physical_begin2d: {physical_begin2d(kpt2d)}")    
    logging.info(f"Headpose: physical_begin: {physical_begin(kpt)}")     
    return X.shape[0] 

def copy_frame_begin(i,X):
    for m in range(i):
        X[m]=X[i]

def compute_KCS_matrix(X):
    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    adjacent_matrix=np.zeros((17,16))
    for i in np.arange(len(I)):
        adjacent_matrix[I[i],i]=1
        adjacent_matrix[J[i],i]=-1
    B=np.einsum("ijk,kn->ijn",X,adjacent_matrix)#[420,3,16]
    
    
    BT=np.transpose(B, (0,2,1))
    
    #pdb.set_trace()
    
    
    KCS_matrix= np.einsum("ijk,ikn->ijn", BT, B)
    return KCS_matrix
    
def find_bone_length(x):
    leg=[2,4,3,5]
    arm=[10,11,12,13]
    leg_length=0
    arm_length=0
    for i in range(len(leg)):
        leg_length+=x[leg[i]][leg[i]]
        
        arm_length+=x[arm[i]][arm[i]]
  
    if leg_length<arm_length:
        return False
    else:
        return True

'''检测2D姿态物理信息'''
import math
import numpy as np

# def check_2D_Physical(X):
#     if X[1,1]
def check_validPhysi2D(X):
    if X[1,2]<X[1,0] and X[1,5]<X[1,0] and X[1,3]<X[1,0] or  X[1,2]<X[1,0] and X[1,5]<X[1,0]  and X[1,6]<X[1,0]: #X[2,2]>X[2,0] and X[2,5]>X[2,0]  and X[2,3]>X[2,0]and X[2,6]>X[2,0]
        return False #输出的为False
    else:
        return True #输出的为True 
def check_valid_length2D(X,threshold=10):
    left_leg=math.fabs(X[1,3]-X[1,2])+math.fabs(X[1,2]-X[1,1])
    right_leg=math.fabs(X[1,6]-X[1,5])+math.fabs(X[1,5]-X[1,4])
    if left_leg<threshold or right_leg<threshold:
        return False
    else:
        return True
def check_validPose2D(X):
    if check_validPhysi2D(X) and check_valid_length2D(X):
        return True  #两个条件都满足时才是正确的姿态
    else:
        return False



'''检测时序信息姿态'''
import numpy as np
import torch
import math
def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))
'''选择跳帧点，把不是的删掉, 跳帧即为下一次摄像机视角变换的起始帧'''
def select_hop_frame(kcs_matrix_2d,X,i,neighbor_threshold=100,continue_threshold=250):
    loss_pre=p_mpjpe(X[i-1].reshape(1,17,3), X[i].reshape(1,17,3)).item() * 1000.0
    loss_aft=p_mpjpe(X[i+1].reshape(1,17,3), X[i].reshape(1,17,3)).item() * 1000.0
    
    loss_contiune=p_mpjpe(X[i+10].reshape(1,17,3), X[i].reshape(1,17,3)).item() * 1000.0
    
    if loss_pre>neighbor_threshold:
        if loss_aft<neighbor_threshold:

            if  loss_contiune<continue_threshold:
                if compute_bone_length(kcs_matrix_2d[i])/1000>400:
                    hop_frame=False #噪声帧
                    end_noisy=select_end_noisy(i,X,kcs_matrix_2d)
                    copy_frame(i,end_noisy,X)
                else:

                    hop_frame=True
            elif loss_contiune>continue_threshold:
                hop_frame=False #不是跳帧
                end_noisy=select_end_noisy_near(i,X,continue_threshold)
                copy_frame(i,end_noisy,X)
        elif loss_aft>neighbor_threshold:
            X[i]=X[i-1]
# def find_near(i,x):
#     min_loss=0
#     for j in range(1,11):
#         min_loss=p_mpjpe(kpts_all[i-1].reshape(1,17,3), kpts_all[i-1+j].reshape(1,17,3)).item() * 1000.0


def select_end_noisy_near(i,X,continue_threshold):#
    for j in range(0,10):
        loss=p_mpjpe(X[i].reshape(1,17,3), X[i+10-j].reshape(1,17,3)).item() * 1000.0
        if loss<continue_threshold:
            return  i+j
# def select_end_noisy_near(i,X,continue_threshold):#
#     for j in range(0,10):
#         loss=p_mpjpe(X[i].reshape(1,17,3), kpts_all[i+10-j].reshape(1,17,3)).item() * 1000.0
#         if loss<continue_threshold:
#             return  i+j

def select_end_noisy(i,X,kcs_matrix_2d):#
    for j in range(0,10):
        
        if compute_bone_length(kcs_matrix_2d[i+j])/1000<400:
            return  i+j
    return i+j
def copy_frame(i,end_noisy,X):
    for m in range(i,end_noisy+1):
        X[m]=X[i-1]#X[end_noisy+1]#用最前面的帧代替
def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))
def mpjpe_all(X,start,end):
    loss_all=[]
    loss=0
    for i in range(start+1,end+1):
        loss+=mpjpe(torch.tensor(X[i-1].reshape(1,17,3)),  torch.tensor(X[i].reshape(1,17,3))).item() * 1000.0
        loss_all.append(mpjpe(torch.tensor(X[i-1].reshape(1,17,3)),  torch.tensor(X[i].reshape(1,17,3))).item() * 1000.0)
    return loss


''''2D bone length'''
def compute_KCS_matrix(X):
    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    adjacent_matrix=np.zeros((17,16))
    for i in np.arange(len(I)):
        adjacent_matrix[I[i],i]=1
        adjacent_matrix[J[i],i]=-1
    B=np.einsum("ijk,kn->ijn",X,adjacent_matrix)#[420,2,16]
    
    
    BT=np.transpose(B, (0,2,1))
    
    #pdb.set_trace()
    
    
    KCS_matrix= np.einsum("ijk,ikn->ijn", BT, B)
    return KCS_matrix

def sym_loss(x):
    left=[3,5,12,13]#[1,3,5,9,12,13]
    right=[2,4,10,11]#[0,2,4,8,10,11]
    loss_sym=0
    for i in range(len(left)):
        loss=math.fabs(x[left[i]][left[i]]-x[right[i]][right[i]])
        #print(loss)
        loss_sym+=loss
        
    return loss_sym
def compute_bone_length(x):
    length=0
    for i in range(16):
        length+=x[i][i]

    return(length)
def find_bone_length(x):
    leg=[2,4,3,5]
    arm=[10,11,12,13]
    leg_length=0
    arm_length=0
    for i in range(len(leg)):
        leg_length+=x[leg[i]][leg[i]]
        
        arm_length+=x[arm[i]][arm[i]]
    # print(leg_length)
    # print(arm_length)
    if leg_length<arm_length:
        return False
    else:
        return True
def compute_bone_length(x):
    length=0
    for i in range(16):
        length+=x[i][i]
    return length


'''检测身体姿态函数'''
import numpy as np
import pdb
import math
import os

'''输入为 2,17'''
def compute_KCS_matrix(X):
    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    adjacent_matrix=np.zeros((17,16))
    for i in np.arange(len(I)):
        adjacent_matrix[I[i],i]=1
        adjacent_matrix[J[i],i]=-1
    B=np.einsum("ijk,kn->ijn",X,adjacent_matrix)#[420,3,16]
    
    
    BT=np.transpose(B, (0,2,1))
    
    #pdb.set_trace()
    
    
    KCS_matrix= np.einsum("ijk,ikn->ijn", BT, B)
    return KCS_matrix

def sym_loss(x):
    left=[1,3,5,9,12,13]
    right=[0,2,4,8,10,11]
    loss_sym=0
    for i in range(len(left)):
        loss=math.fabs(x[left[i]][left[i]]-x[right[i]][right[i]])
        #print(loss)
        loss_sym+=loss
        
    return loss_sym
'''计算Knee膝盖,shoulder肩膀与中心点的合理程度'''
# def check_validPose(X):
#     if X[2,2]<0 and X[2,5]<0  and X[2,3]<0 and X[2,6]<0: #X[2,14]>0 and X[2,11]>0
#         return False
#     else:
#         return True   
def check_validPose(X):
    if X[2,2]>X[2,0] and X[2,5]>X[2,0] and X[2,3]>X[2,0] or  X[2,2]>X[2,0] and X[2,5]>X[2,0]  and X[2,6]>X[2,0]: #X[2,2]>X[2,0] and X[2,5]>X[2,0]  and X[2,3]>X[2,0]and X[2,6]>X[2,0]
        return False #输出的为False
    else:
        return True #输出的为True 
# def check_validPose2D(X):
#     if X[1,2]>X[1,0] and X[1,5]>X[1,0] and X[1,3]>X[1,0] or  X[1,2]>X[1,0] and X[1,5]>X[1,0]  and X[1,6]>X[1,0]: #X[2,2]>X[2,0] and X[2,5]>X[2,0]  and X[2,3]>X[2,0]and X[2,6]>X[2,0]
#         return False #输出的为False
#     else:
#         return True #输出的为True 

def find_valid_pose(kpts_all,valid_pre,leng):
    for num in range(valid_pre+2,leng):
        
        if check_validPose(kpts_all[num]):
            return num
    num=valid_pre
    #print("no valid pose in the end")
    return num
def physical_constraint( begin_num,kpts_all,kpts_all_2D):
    out_put=[]
    i=begin_num
    #abnormal=[]
    valid_pre=-1
    for num in range(0,i):

        out_put.append(np.transpose(kpts_all[i],(1,0)))
    while i <kpts_all.shape[0]:
          
        
        if check_validPose(kpts_all[i]):
            valid_pre=i
            out_put.append(np.transpose(kpts_all[i],(1,0)))
            i+=1
            

           
        else:
            out_put.append(np.transpose(kpts_all[valid_pre],(1,0)))
           
            i=i+1
            # j=find_valid_pose(kpts_all,valid_pre,leng)#选择下一个valid pose
            # if j==valid_pre:##全部用vaild_pre填充
            #     for n in range(valid_pre+1,leng):
                    
            #         out_put.append(kpts_all[j])
            #         i+=1
                
            # else:
            #     for m in range(valid_pre+1,j):
                    
            #         out_put.append(kpts_all[j])
            #         i+=1
    return out_put


def refine(keypoints_2d,keypoints_3d):
    """refine 与直接运行逻辑相同，但是可以作为向外提供的库函数使用，方便调用

    Args:
        keypoint_2d (numpy array): 2D关键点数据,shape=(frame,17,2),COCO标注
        keypoint_3d (numpy array): 3D关键点数据,shape=(frame,17,3)
        
    Return:
        output(numpy array): 提纯后的3D关键点数据,shape=(frame,17,3)
    """
    try:
        keypoints_3d=np.transpose(keypoints_3d, (0,2,1))#-1,3,17
        kcs_3d_matrix=compute_KCS_matrix(keypoints_3d)

        keypoints_2d,_=coco_h36m(keypoints_2d)
        keypoints_2d=np.transpose(keypoints_2d, (0,2,1)) #-1,2,17
        kcs_2d_matrix=compute_KCS_matrix(keypoints_2d)
    except ValueError as e:
        logging.fatal(e)
        logging.fatal(f"Keypoints_3d shape:{keypoints_3d.shape}")
        logging.fatal(f"Keypoints_2d shape:{keypoints_2d.shape}")
        exit(1)
    
    #-------------1. 检测头部及身体位姿是否是正确姿态-----------------------
    begin_frame=find_headPose( keypoints_2d,keypoints_3d,kcs_3d_matrix,kcs_2d_matrix) # 人体姿态为站立姿势的第一帧
    if begin_frame>0: #发现跳帧
        logging.info(f"Begin frame ({begin_frame}) is not origin ")
        copy_frame_begin(begin_frame,keypoints_3d)
        
    #-------------2. 检测2D关键点的身体限制------------------------------
    valid_2d_pre=begin_frame # valid_2d_pre 即先前有效的2D姿态的最近一帧，用于不有效时填充失效帧，
    for i in range(begin_frame,keypoints_2d.shape[0]):
        if check_validPose2D(keypoints_2d[i]):
            valid_2d_pre=i 
        else:
            # 做复制？
            keypoints_3d[i]=keypoints_3d[valid_2d_pre]
    
    #------------ 3. 检测时序信息--------------------------------
    keypoints_3d=np.transpose(keypoints_3d, (0,2,1))#(4250, 17, 3)
    keypoints_2d=np.transpose(keypoints_2d, (0,2,1)) #(4250, 17, 2)
    for i in range(1,keypoints_3d.shape[0]):
        if i+1<keypoints_3d.shape[0] and i+10<keypoints_3d.shape[0]-1:
            select_hop_frame(kcs_2d_matrix,keypoints_3d,i,neighbor_threshold=100,continue_threshold=250)
        elif i<keypoints_3d.shape[0] and i+10>keypoints_3d.shape[0]-1:
            if p_mpjpe(keypoints_3d[i-1].reshape(1,17,3), keypoints_3d[i].reshape(1,17,3)).item() * 1000.0>100:
                copy_frame(i,keypoints_3d.shape[0]-1,keypoints_3d)  #如果最后有突变，一定是噪声,用突变前的一帧替换
    
    #----------- 4. 检测身体姿态---------------------------------
    keypoints_3d=np.transpose(keypoints_3d, (0,2,1))#(4250, 3, 17)
    keypoints_2d=np.transpose(keypoints_2d, (0,2,1))#(4250, 2, 17)
    output=physical_constraint( begin_frame,keypoints_3d,keypoints_2d)
    
    return output
    

if __name__ == '__main__':
    begin=True
    kpts_all=np.load('3d.npz', allow_pickle=True)['reconstruction']#./dataset/3d1.npz
    kpts_all=np.transpose(kpts_all, (0,2,1))#-1,3,17
    kcs_matrix=compute_KCS_matrix(kpts_all)
    
    kpts_all_2D=np.load('keypoints.npz', allow_pickle=True)['reconstruction'] #./dataset/keypoints_1.npz
    kpts_all_2D=kpts_all_2D.squeeze(0)
    kpts_all_2D=np.transpose(kpts_all_2D, (0,2,1)) #-1,2,17
    kcs_matrix2D=compute_KCS_matrix(kpts_all_2D)
    
    '''检测头部及身体位姿是否是正确姿态'''

   
    #print(find_bone_length(kcs_matrix[58]))
    
    if begin==True:
        begin_num=find_headPose( kpts_all_2D,kpts_all,kcs_matrix,kcs_matrix2D)
        begin=False
    if  begin_num>0:
        print(begin_num)
        copy_frame_begin(begin_num,kpts_all)
    '''检测2D关键点的身体限制'''
    valid_2d_pre=begin_num
    for i in range(begin_num,kpts_all_2D.shape[0]):
        # X=kpts_all_2D[i]
        if check_validPose2D(kpts_all_2D[i]):
            valid_2d_pre=i
           
        else:
            kpts_all[i]=kpts_all[valid_2d_pre]
            #print(i)

            # X=np.transpose(X,(1,0))
            #print(X.shape)
            # plt_2D_single(X)
            # #print(kpts_all[4107])
            # plt.show()
            #print(i)
    
    '''检测时序信息'''
    kpts_all=np.transpose(kpts_all, (0,2,1))#(4250, 17, 3)
    kpts_all_2D=np.transpose(kpts_all_2D, (0,2,1)) #(4250, 17, 2)
    for i in range(1,kpts_all.shape[0]):
        if i+1<kpts_all.shape[0] and i+10<kpts_all.shape[0]-1:
            select_hop_frame(kcs_matrix2D,kpts_all,i,neighbor_threshold=100,continue_threshold=250)
        elif i<kpts_all.shape[0] and i+10>kpts_all.shape[0]-1:
            if p_mpjpe(kpts_all[i-1].reshape(1,17,3), kpts_all[i].reshape(1,17,3)).item() * 1000.0>100:
                copy_frame(i,kpts_all.shape[0]-1,kpts_all)  #如果最后有突变，一定是噪声,用突变前的一帧替换
    
    '''检测身体姿态'''
    kpts_all=np.transpose(kpts_all, (0,2,1))#(4250, 3, 17)
    kpts_all_2D=np.transpose(kpts_all_2D, (0,2,1))#(4250, 2, 17)
    out_put=physical_constraint( begin_num,kpts_all,kpts_all_2D)
   
    print(len(out_put))
    output_3d_path='output_3D/'
    output_npz_3D ='output_3D/'+ 'refine_3d'#+'_{}'.format(i)
    #print(pose_out_all.shape)
    os.makedirs(output_3d_path, exist_ok=True)
    np.savez_compressed(output_npz_3D, reconstruction=out_put)# pose_out_no_process


   