import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
import roslib
from math import pi, atan, tan, sin, cos
from math import degrees as deg
from math import radians as rad
import rospy
import os
from sensor_msgs.msg import Image
from std_msgs.msg import String
# from robotiq_85_msgs.msg import GripperCmd, GripperStat
# import urx
from mask_rcnn_ros.msg import *
from scipy import ndimage
from singleshot import singleshot

class UR5_UI(QDialog):

	def __init__(self):
		super(UR5_UI,self).__init__()
		loadUi('UR_Robot_singleshot.ui',self)

		self.setWindowTitle('UR5_UI')
		# self.ur5 = urx.Robot("192.168.1.12")

		# self.robotiqPub = rospy.Publisher('/gripper/cmd', GripperCmd, queue_size=10)

		# rospy.Subscriber('/gripper/stat', GripperStat, self.update_stat, queue_size=10)
		
		datacfg     = "/home/david/Downloads/singleshotpose-master/cfg/diget.data"
		cfgfile     = "/home/david/Downloads/singleshotpose-master/cfg/yolo-pose.cfg"
		weightfile  = "/home/david/Downloads/singleshotpose-master/backup/diget/model_diget.weights"

		self.bridge = CvBridge()
		self.pose_estimator = singleshot(datacfg, cfgfile, weightfile)

		rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback_rgb)

		# self.result = Result()
		self.frame = None		
		self.box_3d_color = (238,198,10)
		self.box_3d_color1 = (246,0,60)

		############ Button Connection ############
		self.Streaming_Start_Btn.clicked.connect(self.start)
		self.Pose1_Btn.clicked.connect(self.pose1)
		self.Home_Btn.clicked.connect(self.home)
		self.Grasp_Btn.clicked.connect(self.grasp)
		###########################################

		############## Task Position ##############
		self.lin_accel = 0.5
		self.lin_vel = 0.4
		self.init_wpr = (0.400, 0.211, 0.200, 2.20, -2.20, 0)
		###########################################

	def home(self):
		print("home")
		# self.ur5.movel(self.init_wpr, self.lin_accel, self.lin_vel)
		# rospy.sleep(1)

	def pose1(self):		
		start_time = time.time()
		self.estimation_result_img, self.R_pr, self.t_pr = self.pose_estimator.predict(self.frame_rgb,self.box_3d_color,self.box_3d_color1)
		print("time : %f"%(time.time() - start_time))
		self.result_monitor(self.estimation_result_img)

	def grasp(self):
		print("grasp")


	def callback_result(self,data):
		self.result = data


	def callback_rgb(self, data):
		try:
			self.frame = self.bridge.imgmsg_to_cv2(data,"bgr8")
			self.frame_rgb = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
		except CvBridgeError as e:
			print(e)


	def start(self):
		self.timer = QTimer()
		self.timer.timeout.connect(self.streaming_start)
		self.timer.start(1000/60)

	def streaming_start(self):
		frame = cv2.resize(self.frame_rgb,(384,288))
		self.image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
		self.pixmapImage = QtGui.QPixmap.fromImage(self.image)
		self.Video_Streaming.setPixmap(self.pixmapImage)
		self.pose1()

	def result_monitor(self, img):
		result = img
		result = cv2.resize(result,(384,288))
		self.result_image = QtGui.QImage(result, result.shape[1], result.shape[0], QImage.Format_RGB888)
		self.pixmapImage_result = QtGui.QPixmap.fromImage(self.result_image)
		self.Video_Streaming_Result.setPixmap(self.pixmapImage_result)


def main(args):
	rospy.init_node('UR5_UI', anonymous=True)

if __name__=='__main__':
	main(sys.argv)
	app = QApplication(sys.argv)
	widget = UR5_UI()
	widget.show()
	sys.exit(app.exec_())
