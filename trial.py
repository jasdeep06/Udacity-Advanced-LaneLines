import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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

    elif color_space=="grayscale":
        return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)


    else:
        print("Incorrect color space.Please choose one from HLS,HSV,grayscale or RGB")
        return 0

image=mpimg.imread("output_images/undistorted_road/test4.jpg")
r_channel=get_channeled_image(image,"RGB","R")
g_channel=get_channeled_image(image,"RGB","G")
b_channel=get_channeled_image(image,"RGB","B")
h_channel_hsv=get_channeled_image(image,"HSV","H")
s_channel_hsv=get_channeled_image(image,"HSV","S")
v_channel_hsv=get_channeled_image(image,"HSV","V")
h_channel_hls=get_channeled_image(image,"HLS","H")
l_channel_hls=get_channeled_image(image,"HLS","L")
#s_channel_hls=get_channeled_image(image,"HLS","S")
hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
s_channel_hls=hls[:,:,2]
plt.imshow(r_channel)
plt.show()
plt.imshow(g_channel)
plt.show()
plt.imshow(b_channel)
plt.show()
plt.imshow(h_channel_hsv)
plt.show()
plt.imshow(s_channel_hsv)
plt.show()
plt.imshow(v_channel_hsv)
plt.show()
plt.imshow(h_channel_hls)
plt.show()
plt.imshow(l_channel_hls)
plt.show()
plt.imshow(s_channel_hls,cmap="gray")
plt.show()
