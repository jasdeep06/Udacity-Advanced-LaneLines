import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import os
def get_channeled_image(image,color_space,channel=""):

    if color_space=="HLS":
        color_spaced_image=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)

        if channel=="H":
            channeled_image=color_spaced_image[:,:,0]
            return channeled_image
        elif channel=="L":
            channeled_image=color_spaced_image[:,:,1]
            return channeled_image

        elif channel=="S":
            channeled_image=color_spaced_image[:,:,2]
            return channeled_image

        elif channel=="all":
            channeled_image=color_spaced_image
            return channeled_image

        else:
            print("Please enter a valid channel (H,L or S)")
            return 0

    elif color_space=="HSV":
        color_spaced_image=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

        if channel=="H":
            channeled_image=color_spaced_image[:,:,0]
            return channeled_image

        elif channel=="S":
            channeled_image=color_spaced_image[:,:,1]
            return channeled_image

        elif channel=="V":
            channeled_image=color_spaced_image[:,:,2]
            return channeled_image

        elif channel=="all":
            channeled_image=color_spaced_image
            return channeled_image

        else:
            print("Please enter a valid channel (H,S or V)")
            return 0

    elif color_space=="RGB":
        color_spaced_image=image

        if channel=="R":
            channeled_image=color_spaced_image[:,:,0]
            return channeled_image

        elif channel=="G":
            channeled_image=color_spaced_image[:,:,1]
            return channeled_image

        elif channel=="B":
            channeled_image=color_spaced_image[:,:,2]
            return channeled_image

        elif channel=="all":
            channeled_image=color_spaced_image
            return channeled_image

        else:
            print("Please enter a valid channel (R,G or B)")
            return 0

    elif color_space=="LUV":
        color_spaced_image=cv2.cvtColor(image,cv2.COLOR_RGB2LUV)

        if channel == "L":
            channeled_image = color_spaced_image[:, :, 0]
            return channeled_image

        elif channel == "U":
            channeled_image = color_spaced_image[:, :, 1]
            return channeled_image

        elif channel == "V":
            channeled_image = color_spaced_image[:, :, 2]
            return channeled_image

        elif channel == "all":
            channeled_image = color_spaced_image
            return channeled_image

        else:
            print("Please enter a valid channel (L,U or V)")
            return 0


    elif color_space=="LAB":
        color_spaced_image=cv2.cvtColor(image,cv2.COLOR_RGB2Lab)

        if channel == "L":
            channeled_image = color_spaced_image[:, :, 0]
            return channeled_image

        elif channel == "A":
            channeled_image = color_spaced_image[:, :, 1]
            return channeled_image

        elif channel == "B":
            channeled_image = color_spaced_image[:, :, 2]
            return channeled_image

        elif channel == "all":
            channeled_image = color_spaced_image
            return channeled_image

        else:
            print("Please enter a valid channel (L,a or b)")
            return 0


    elif color_space=="grayscale":
        return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)




    else:
        print("Incorrect color space.Please choose one from HLS,HSV,grayscale or RGB")
        return 0


def gradient_thresholding_mag(channeled_image,kernel_size,thresh_min_mag,thresh_max_mag):

    sobelx=cv2.Sobel(channeled_image,cv2.CV_64F,1,0,ksize=kernel_size)

    sobely=cv2.Sobel(channeled_image,cv2.CV_64F,0,1,ksize=kernel_size)

    sobel_mag=np.sqrt(sobelx**2+sobely**2)

    scaled_sobel_mag=np.uint8(255*(sobel_mag)/np.max(sobel_mag))

    binary=np.zeros_like(scaled_sobel_mag)

    if scaled_sobel_mag.shape[-1]==3:
        binary[(scaled_sobel_mag > thresh_min_mag ) & (scaled_sobel_mag < thresh_max_mag)]=255
    else:
        binary[(scaled_sobel_mag > thresh_min_mag ) & (scaled_sobel_mag <= thresh_max_mag)]=1



    return binary




def gradient_thresholding_dir(channeled_image,kernel_size,thresh_min_dir,thresh_max_dir):
    sobelx = cv2.Sobel(channeled_image, cv2.CV_64F, 1, 0, ksize=kernel_size)

    sobely = cv2.Sobel(channeled_image, cv2.CV_64F, 0, 1, ksize=kernel_size)



    sobel_dir = np.arctan2(abs(sobely),abs(sobelx))


    binary = np.zeros_like(sobel_dir)

    if sobel_dir.shape[-1] == 3:
        binary[(sobel_dir > thresh_min_dir) & (sobel_dir < thresh_max_dir)] = 255
    else:
        binary[(sobel_dir > thresh_min_dir) & (sobel_dir < thresh_max_dir)] = 1

    return binary

def gradient_thresholding_x(channeled_image,kernel_size,thresh_min_dir,thresh_max_dir):
    sobelx = cv2.Sobel(channeled_image, cv2.CV_64F, 1, 0, ksize=kernel_size)




    abs_sobel=abs(sobelx)

    scaled_sobel_abs=np.uint8(255*(abs_sobel)/np.max(abs_sobel))

    binary = np.zeros_like(scaled_sobel_abs)

    if scaled_sobel_abs.shape[-1] == 3:
        binary[(scaled_sobel_abs > thresh_min_dir) & (scaled_sobel_abs < thresh_max_dir)] = 255
    else:
        binary[(scaled_sobel_abs > thresh_min_dir) & (scaled_sobel_abs < thresh_max_dir)] = 1

    return binary



def color_thresholding(channeled_image,thresh_min_col,thresh_max_col):

    binary=np.zeros_like(channeled_image)

    binary[(channeled_image > thresh_min_col) & (channeled_image < thresh_max_col)] = 1

    return binary

def combine_thresholds(binary_mag,binary_dir,binary_col):

    combined_binary=np.zeros_like(binary_mag)

    combined_binary[((binary_mag==1) & (binary_dir==1)) | (binary_col == 1)]=1

    return combined_binary



def thresholding(image,color_space,channel,thresh_min_mag,thresh_max_mag,thresh_min_dir,thresh_max_dir,thresh_min_col,thresh_max_col):

    channeled_image=get_channeled_image(image,color_space,channel)

    binary_mag=gradient_thresholding_mag(channeled_image,3,thresh_min_mag,thresh_max_mag)
    binary_dir=gradient_thresholding_dir(channeled_image,20,thresh_min_dir,thresh_max_dir)
    binary_col=color_thresholding(channeled_image,thresh_min_col,thresh_max_col)
    combined_binary=combine_thresholds(binary_mag,binary_dir,binary_col)
    #plt.imshow(binary_dir,cmap="binary")
    plot(image,channeled_image,binary_mag,binary_dir,binary_col,combined_binary,color_space,channel,thresh_min_mag,thresh_max_mag,thresh_min_dir,thresh_max_dir,thresh_min_col,thresh_max_col)


def plot(image,channeled_image,binary_mag,binary_dir,binary_col,combined_binary,color_space,channel,thresh_min_mag,thresh_max_mag,thresh_min_dir,thresh_max_dir,thresh_min_col,thresh_max_col):
    fig = plt.figure(figsize=(14, 7))
    ax0=fig.add_subplot(3,2,1)
    ax0.set_title("Original Image "+color_space)
    plt.imshow(image)
    ax1=fig.add_subplot(3,2,2)
    ax1.set_title("channel "+channel)
    plt.imshow(channeled_image)
    ax2=fig.add_subplot(3,2,3)
    ax2.set_title("Gradient Magnitude ( "+str(thresh_min_mag)+","+str(thresh_max_mag)+")")
    plt.imshow(binary_mag,cmap="gray")
    ax2 = fig.add_subplot(3, 2, 4)
    ax2.set_title("Gradient Direction ( " + str(thresh_min_dir) + "," + str(thresh_max_dir) + ")")
    plt.imshow(binary_dir,cmap="gray")
    ax2 = fig.add_subplot(3, 2, 5)
    ax2.set_title(" Color ( " + str(thresh_min_col) + "," + str(thresh_max_col) + ")")
    plt.imshow(binary_col,cmap="gray")
    ax2 = fig.add_subplot(3, 2, 6)
    ax2.set_title("Combined")
    plt.imshow(combined_binary,cmap="gray")
    fig.tight_layout()


def sample_randomly_int(sampling_range):
    return random.randint(sampling_range[0],sampling_range[1])

def sample_randomly(sampling_range):
    return random.uniform(sampling_range[0],sampling_range[1])
"""
num_cases=10
image_addresses=glob.glob("output_images/undistorted_road/*")
for image_address in image_addresses:
    print(image_address)
    image=mpimg.imread(image_address)
    color_spaces_list=["RGB","HLS","HSV"]
    for space in color_spaces_list:
        k=0
        for i in range(len(space)):
            for j in range(num_cases):
                k=k+1
                thresholding(image,space,space[i],sample_randomly_int((10,100)),sample_randomly_int((150,250)),sample_randomly((0.5,1)),sample_randomly((1,1.5)),sample_randomly_int((10,100)),sample_randomly_int((150,250)))
                plt.savefig("output_images/thresholding/" +
                        os.path.basename(str(image_address)).split(".")[0] + "/random/" +space+"/"+str(k))
                print("output_images/thresholding/" +
                        os.path.basename(str(image_address)).split(".")[0] + "/random/" +space+"/"+str(k))
                plt.close()

"""
image_addresses=glob.glob("output_images/undistorted_road/*")


for image_address in image_addresses:

    image=mpimg.imread(image_address)
    grayscale=get_channeled_image(image,"grayscale")
    s_channel=get_channeled_image(image,"HLS","S")
    r_channel=get_channeled_image(image,"RGB","R")
    l_channel=get_channeled_image(image,"LUV","L")
    b_channel=get_channeled_image(image,"LAB","B")
    h_channel=get_channeled_image(image,"HSV","H")


    gt=gradient_thresholding_mag(grayscale,3,50,255)
    gd=gradient_thresholding_dir(grayscale,3,0.7,1.3)

    cts=color_thresholding(s_channel,170,255)
    ctl=color_thresholding(l_channel,225,255)
    ctb=color_thresholding(b_channel,155,200)
    cth=color_thresholding(h_channel,19,40)
    combined_binary=np.zeros_like(cts)
    combined_binary_col=np.zeros_like(cts)


    combined_binary[((ctb==1)|(cth==1)|(ctl==1) | (gt==1) &(gd==1)) ]=1

    plt.imsave("output_images/binary_road/"+os.path.basename(image_address),np.array(combined_binary).reshape(combined_binary.shape[0],combined_binary.shape[1]),cmap="gray", vmin=0, vmax=1)

    #plt.imshow(combined_binary,cmap="gray", vmin=0, vmax=1)



    #plt.show()

"""
def combination_thresholding(image,thresh_min_mag, thresh_max_mag,thresh_min_dir, thresh_max_dir,thresh_min_s,thresh_max_s,thresh_min_r,thresh_max_r):
    s_channel = get_channeled_image(image, "HLS", "S")
    r_channel = get_channeled_image(image, "RGB", "R")

    s_channel = cv2.GaussianBlur(s_channel, (3, 3), 0)
    r_channel = cv2.GaussianBlur(r_channel, (3, 3), 0)

    gt = gradient_thresholding_mag(s_channel, 3, thresh_min_mag, thresh_max_mag)
    gd = gradient_thresholding_dir(s_channel, 3, thresh_min_dir, thresh_max_dir)

    cts = color_thresholding(s_channel, thresh_min_s, thresh_max_s)
    ctr = color_thresholding(r_channel, thresh_min_r,thresh_max_r)
    combined_binary = np.zeros_like(cts)

    combined_binary[((cts == 1) & (ctr == 1)) | (gt == 1) & (gd == 1)] = 1

    fig = plt.figure(figsize=(20, 10))
    ax0 = fig.add_subplot(3, 3, 1)
    ax0.set_title("Original Image")
    plt.imshow(image)
    #ax1 = fig.add_subplot(4, 2, 2)
    #ax1.set_title("S channel")
    #plt.imshow(s_channel)
    #ax1 = fig.add_subplot(4, 2, 3)
    #ax1.set_title("R channel")
    #plt.imshow(r_channel)
    ax1 = fig.add_subplot(3, 3, 2)
    ax1.set_title("Gradient Magnitude ( " + str(thresh_min_mag) + "," + str(thresh_max_mag) + ")")
    plt.imshow(gt, cmap="gray")
    ax1 = fig.add_subplot(3, 3, 3)
    ax1.set_title("Gradient Direction ( " + str(thresh_min_dir) + "," + str(thresh_max_dir) + ")")
    plt.imshow(gd, cmap="gray")
    ax1 = fig.add_subplot(3, 3, 4)
    ax1.set_title("S Color ( " + str(thresh_min_s) + "," + str(thresh_max_s) + ")")
    plt.imshow(cts, cmap="gray")
    ax1 = fig.add_subplot(3, 3, 5)
    ax1.set_title("R Color ( " + str(thresh_min_r) + "," + str(thresh_max_r) + ")")
    plt.imshow(ctr, cmap="gray")
    ax1 = fig.add_subplot(3, 3, 6)
    ax1.set_title("Combined")
    plt.imshow(combined_binary, cmap="gray")
    fig.tight_layout()


num_cases=10
image_addresses=glob.glob("output_images/undistorted_road/*")
for image_address in image_addresses:
    k=0
    image=mpimg.imread(image_address)

    for cases in range(num_cases):
        k=k+1

        combination_thresholding(image,sample_randomly_int((60,90)),sample_randomly_int((210,250)),sample_randomly((0.7,1)),sample_randomly((1.01,1.3)),sample_randomly_int((70,100)),sample_randomly_int((220,255)),sample_randomly_int((180,210)),sample_randomly_int((220,255)))
        plt.savefig("output_images/thresholding/" +
                        os.path.basename(str(image_address)).split(".")[0] + "/precise/"+str(k))
        print("output_images/thresholding/" +
                        os.path.basename(str(image_address)).split(".")[0] + "/precise/"+str(k))
        plt.close()

"""

