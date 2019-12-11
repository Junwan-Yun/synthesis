import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
warnings.filterwarnings("ignore")

from darknet import Darknet
from utils import *
from MeshPly import MeshPly
from PIL import Image
import cv2
import PyKDL

class singleshot(object):
    def __init__(self, datacfg, cfgfile, weightfile):

        meshname  = "/home/david/Downloads/singleshotpose-master/CUSTOM/cleanser/diget.ply"

        seed         = int(time.time())
        gpus         = '0'     # Specify which gpus to use
        self.test_width   = 544
        self.test_height  = 544

        torch.manual_seed(seed)
        self.use_cuda = True
        if self.use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            torch.cuda.manual_seed(seed)

        self.num_classes     = 1
        self.eps             = 1e-5
        self.conf_thresh     = 0.1
        self.nms_thresh      = 0.4
        self.match_thresh    = 0.5

        mesh               = MeshPly(meshname)
        vertices           = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
        self.corners3D     = get_3D_corners(vertices)

        self.internal_calibration = get_camera_intrinsic()

        self.model = Darknet(cfgfile)
        self.model.print_network()
        self.model.load_weights(weightfile)
        self.model.cuda()
        self.model.eval()

        # self.preds_trans         = []
        # self.preds_rot           = []
        # self.preds_corners2D     = []
        # self.gts_trans           = []
        # self.gts_rot             = []
        # self.gts_corners2D       = []


    def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i

    def predict(self, img, box_3d_color, box_3d_color1):
        ori_img = img
        img = cv2.resize(img,(self.test_width,self.test_height))
        trans = transforms.Compose([transforms.ToTensor(),])
        data = trans(img)
        data = data.resize_((1,data.shape[0],data.shape[1],data.shape[2]))
        # Pass data to GPU
        if self.use_cuda:
            data = data.cuda()
        
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        
        # Forward pass
        output = self.model(data).data

        all_boxes = get_region_boxes(output, self.conf_thresh, self.num_classes)

        for i in range(output.size(0)):
        
            # For each image, get all the predictions
            boxes   = all_boxes[i]
        
            best_conf_est = -1
            # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
            for j in range(len(boxes)):
                if (boxes[j][18] > best_conf_est):
                    box_pr        = boxes[j]
                    best_conf_est = boxes[j][18]

            # Denormalize the corner predictions 
            corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
            corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
            corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480

            R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(self.internal_calibration, dtype='float32'))
            proj_2d_gt = corners2D_pr

            R_tmp = []

            for i in range(3):
                for j in range(3):
                    R_tmp.append(R_pr[i][j])

            rotation = PyKDL.Rotation(R_tmp[0],R_tmp[1],R_tmp[2],R_tmp[3],R_tmp[4],R_tmp[5],R_tmp[6],R_tmp[7],R_tmp[8])
            self.rotation_rpy = rotation.GetRPY()
            print(self.rotation_rpy)


            x1 = int(proj_2d_gt[1][0])
            y1 = int(proj_2d_gt[1][1])
            x2 = int(proj_2d_gt[3][0])
            y2 = int(proj_2d_gt[3][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[5][0])
            y1 = int(proj_2d_gt[5][1])
            x2 = int(proj_2d_gt[7][0])
            y2 = int(proj_2d_gt[7][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[1][0])
            y1 = int(proj_2d_gt[1][1])
            x2 = int(proj_2d_gt[5][0])
            y2 = int(proj_2d_gt[5][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[3][0])
            y1 = int(proj_2d_gt[3][1])
            x2 = int(proj_2d_gt[7][0])
            y2 = int(proj_2d_gt[7][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)                



            x1 = int(proj_2d_gt[1][0])
            y1 = int(proj_2d_gt[1][1])
            x2 = int(proj_2d_gt[2][0])
            y2 = int(proj_2d_gt[2][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[3][0])
            y1 = int(proj_2d_gt[3][1])
            x2 = int(proj_2d_gt[4][0])
            y2 = int(proj_2d_gt[4][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[5][0])
            y1 = int(proj_2d_gt[5][1])
            x2 = int(proj_2d_gt[6][0])
            y2 = int(proj_2d_gt[6][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[7][0])
            y1 = int(proj_2d_gt[7][1])
            x2 = int(proj_2d_gt[8][0])
            y2 = int(proj_2d_gt[8][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)


            x1 = int(proj_2d_gt[2][0])
            y1 = int(proj_2d_gt[2][1])
            x2 = int(proj_2d_gt[4][0])
            y2 = int(proj_2d_gt[4][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[6][0])
            y1 = int(proj_2d_gt[6][1])
            x2 = int(proj_2d_gt[8][0])
            y2 = int(proj_2d_gt[8][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[2][0])
            y1 = int(proj_2d_gt[2][1])
            x2 = int(proj_2d_gt[6][0])
            y2 = int(proj_2d_gt[6][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[4][0])
            y1 = int(proj_2d_gt[4][1])
            x2 = int(proj_2d_gt[8][0])
            y2 = int(proj_2d_gt[8][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

            x1 = int(proj_2d_gt[2][0])
            y1 = int(proj_2d_gt[2][1])
            x2 = int(proj_2d_gt[8][0])
            y2 = int(proj_2d_gt[8][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color1,2)

            x1 = int(proj_2d_gt[4][0])
            y1 = int(proj_2d_gt[4][1])
            x2 = int(proj_2d_gt[6][0])
            y2 = int(proj_2d_gt[6][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color1,2)

            x1 = int(proj_2d_gt[3][0])
            y1 = int(proj_2d_gt[3][1])
            x2 = int(proj_2d_gt[8][0])
            y2 = int(proj_2d_gt[8][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color1,2)

            x1 = int(proj_2d_gt[4][0])
            y1 = int(proj_2d_gt[4][1])
            x2 = int(proj_2d_gt[7][0])
            y2 = int(proj_2d_gt[7][1])
            cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color1,2)


            return ori_img, R_pr, t_pr
