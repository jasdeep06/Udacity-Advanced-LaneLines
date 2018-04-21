import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
#image2=mpimg.imread("output_images/undistorted_road/straight_lines2.jpg")
road_addresses=glob.glob("output_images/undistorted_road/*")

def perspective_matrix():
    image=mpimg.imread("output_images/undistorted_road/straight_lines1.jpg")




    src = np.float32([
        [334, 639],
        [582, 469],
        [711, 469],
        [969, 639]
    ])

    dst = np.float32([
        [300, 717],
        [350, 0],
        [900, 0],
        [950, 717]

    ])

    M=cv2.getPerspectiveTransform(src,dst)

    Minv=cv2.getPerspectiveTransform(dst,src)

    return M,Minv

M,Minv=perspective_matrix()


undistorted_road_addresses=glob.glob("output_images/binary_road/*")

for undistorted_road_address in undistorted_road_addresses:
    image=mpimg.imread(undistorted_road_address)
    print(image.shape)


    image_size = (image.shape[1], image.shape[0])
    print(image_size)

    warped=cv2.warpPerspective(image,M,image_size,flags=cv2.INTER_LINEAR)
    print(warped.shape)
    plt.imsave("output_images/perspective_binary/"+os.path.basename(undistorted_road_address),np.array(warped[:,:,0]).reshape(warped.shape[0],warped.shape[1]),cmap="gray", vmin=0, vmax=1)

