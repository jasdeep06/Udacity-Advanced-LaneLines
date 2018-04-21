import cv2
import numpy as np
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import pickle

images=glob.glob("camera_cal/calibration*.jpg")
road_images=glob.glob("test_images/*")
object_points=[]
image_points=[]
objp=np.zeros((6*9,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)


for image_address in images:
    image=mpimg.imread(image_address)
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,(9,6),None)
    if ret==True:
        cv2.drawChessboardCorners(image,(9,6),corners,ret)
        image_points.append(corners)
        object_points.append(objp)


for image_address in road_images:
    image_op = mpimg.imread(image_address)
    gray = cv2.cvtColor(image_op, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    dst = cv2.undistort(image_op, mtx, dist, None, mtx)
    pickle.dump([ret, mtx, dist, rvecs, tvecs],open("camera_calibration.p","wb"))

    plt.imsave("output_images/undistorted_road/"+os.path.basename(image_address),dst)

