import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2



#picking up 1st channel as imread adds extra channels

binary_warped=mpimg.imread("output_images/perspective_binary/straight_lines1.jpg")
binary_warped=binary_warped[:,:,0]




print(binary_warped.shape)
#histogram of lower half of image
histogram=np.sum(binary_warped[binary_warped.shape[0]//2:,:],axis=0)


out_img = np.dstack((binary_warped, binary_warped, binary_warped))

#midpoint of histogram
midpoint=len(histogram)//2


#left position of maxima
left_base=np.argmax(histogram[0:midpoint])

#right position of maxima
right_base=np.argmax(histogram[midpoint:])+midpoint

nonzero_pixels=binary_warped.nonzero()
nonzero_pixels_x=np.array(nonzero_pixels[1])
nonzero_pixels_y=np.array(nonzero_pixels[0])

number_of_windows=9

height_of_window=np.int(binary_warped.shape[0]//number_of_windows)

#current left and right centres
current_left_centre=left_base
current_right_centre=right_base

all_left_ones=[]
all_right_ones=[]


#width of window
margin=100

minpixels=50

for window in range(number_of_windows):

    left_extreme_x=current_left_centre-margin
    left_sober_x=current_left_centre+margin
    right_extreme_x=current_right_centre+margin
    right_sober_x=current_right_centre-margin
    bottom_y=binary_warped.shape[0]-(window)*height_of_window
    top_y=binary_warped.shape[0]-(window+1)*height_of_window


    cv2.rectangle(out_img,(left_extreme_x,bottom_y),(left_sober_x,top_y),(0,255,0),2)
    cv2.rectangle(out_img,(right_extreme_x,bottom_y),(right_sober_x,top_y),(0,255,0),2)

    left_ones=((left_extreme_x < nonzero_pixels_x) & (left_sober_x > nonzero_pixels_x ) & (top_y<nonzero_pixels_y) & (bottom_y > nonzero_pixels_y )).nonzero()[0]

    right_ones=((right_extreme_x>nonzero_pixels_x) & (right_sober_x<nonzero_pixels_x) & (top_y<nonzero_pixels_y) & (bottom_y>nonzero_pixels_y)).nonzero()[0]


    all_left_ones.append(left_ones)
    all_right_ones.append(right_ones)

    if len(left_ones)>minpixels:
        current_left_centre=np.int(np.mean(nonzero_pixels_x[left_ones]))

    if len(right_ones)>minpixels:
        current_right_centre=np.int(np.mean(nonzero_pixels_x[right_ones]))

all_left_ones=np.concatenate(all_left_ones)
all_right_ones=np.concatenate(all_right_ones)

left_x=nonzero_pixels_x[all_left_ones]
left_y=nonzero_pixels_y[all_left_ones]
right_x=nonzero_pixels_x[all_right_ones]
right_y=nonzero_pixels_y[all_right_ones]

left_fit = np.polyfit(left_y, left_x, 2)
right_fit = np.polyfit(right_y, right_x, 2)

ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzero_pixels_y[all_left_ones], nonzero_pixels_x[all_left_ones]] = [255, 0, 0]
out_img[nonzero_pixels_y[all_right_ones], nonzero_pixels_y[all_right_ones]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()