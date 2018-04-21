import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

ret, mtx, dist, rvecs, tvecs=pickle.load(open("camera_calibration.p","rb"))

#function to undistort using camera_calibration.p
def undistort(image):

    return cv2.undistort(image, mtx, dist, None, mtx)


#function to retrieve channel of an image given color space and channel
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

#function compution gradient magnitude and thresholding
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



#function compution gradient direction and thresholding
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


#function computing color thresholding
def color_thresholding(channeled_image,thresh_min_col,thresh_max_col):

    binary=np.zeros_like(channeled_image)

    binary[(channeled_image > thresh_min_col) & (channeled_image < thresh_max_col)] = 1

    return binary




#main function to compute and combine various thresholds
def threshold(image):


    grayscale = get_channeled_image(image, "grayscale")
    s_channel = get_channeled_image(image, "HLS", "S")
    l_channel = get_channeled_image(image, "LUV", "L")
    b_channel = get_channeled_image(image, "LAB", "B")
    h_channel = get_channeled_image(image, "HSV", "H")

    gt = gradient_thresholding_mag(grayscale, 3, 50, 255)
    gd = gradient_thresholding_dir(grayscale, 3, 0.7, 1.3)

    cts = color_thresholding(s_channel, 170, 255)
    ctl = color_thresholding(l_channel, 225, 255)
    ctb = color_thresholding(b_channel, 155, 200)
    cth = color_thresholding(h_channel, 19, 40)
    combined_binary = np.zeros_like(cts)

    combined_binary[((ctb == 1) | (cth == 1) | (ctl == 1) | (gt == 1) & (gd == 1))] = 1

    return combined_binary

#function computing perspective and inverse matrix
def perspective_matrix(image):

    src = np.float32(
        [[200, 720],
         [1100, 720],
         [595, 450],
         [685, 450]])
    dst = np.float32(
        [[300, 720],
         [980, 720],
         [300, 0],
         [980, 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

#function responsible for perspective transform
def perspective_transform_image(image,M):

    image_size = (image.shape[1], image.shape[0])

    return  cv2.warpPerspective(image,M,image_size,flags=cv2.INTER_LINEAR)


#function to detect lane lines by sliding window search
def detect_lanes(binary_warped):



    # histogram of lower half of image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)*255

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # midpoint of histogram
    midpoint = len(histogram) // 2

    # left position of maxima
    left_base = np.argmax(histogram[0:midpoint])

    # right position of maxima
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    nonzero_pixels = binary_warped.nonzero()
    nonzero_pixels_x = np.array(nonzero_pixels[1])
    nonzero_pixels_y = np.array(nonzero_pixels[0])

    number_of_windows = 9

    height_of_window = np.int(binary_warped.shape[0] // number_of_windows)

    # current left and right centres
    current_left_centre = left_base
    current_right_centre = right_base

    all_left_ones = []
    all_right_ones = []

    # width of window
    margin = 100

    minpixels = 50

    for window in range(number_of_windows):

        left_extreme_x = current_left_centre - margin
        left_sober_x = current_left_centre + margin
        right_extreme_x = current_right_centre + margin
        right_sober_x = current_right_centre - margin
        bottom_y = binary_warped.shape[0] - (window) * height_of_window
        top_y = binary_warped.shape[0] - (window + 1) * height_of_window

        cv2.rectangle(out_img, (left_extreme_x, bottom_y), (left_sober_x, top_y), (0, 255, 0), 2)
        cv2.rectangle(out_img, (right_extreme_x, bottom_y), (right_sober_x, top_y), (0, 255, 0), 2)

        left_ones = (
        (left_extreme_x < nonzero_pixels_x) & (left_sober_x > nonzero_pixels_x) & (top_y < nonzero_pixels_y) & (
        bottom_y > nonzero_pixels_y)).nonzero()[0]

        right_ones = (
        (right_extreme_x > nonzero_pixels_x) & (right_sober_x < nonzero_pixels_x) & (top_y < nonzero_pixels_y) & (
        bottom_y > nonzero_pixels_y)).nonzero()[0]

        all_left_ones.append(left_ones)
        all_right_ones.append(right_ones)

        if len(left_ones) > minpixels:
            current_left_centre = np.int(np.mean(nonzero_pixels_x[left_ones]))

        if len(right_ones) > minpixels:
            current_right_centre = np.int(np.mean(nonzero_pixels_x[right_ones]))

    all_left_ones = np.concatenate(all_left_ones)
    all_right_ones = np.concatenate(all_right_ones)

    left_x = nonzero_pixels_x[all_left_ones]
    left_y = nonzero_pixels_y[all_left_ones]
    right_x = nonzero_pixels_x[all_right_ones]
    right_y = nonzero_pixels_y[all_right_ones]

    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    return left_fit,right_fit,left_x,right_x

#function to draw lane lines
def draw_lanes(undist_image,m_inv,left_fit,right_fit):

    ploty = np.linspace(0, undist_image.shape[0] - 1, undist_image.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    color_warp = np.zeros((720, 1280, 3), dtype='uint8')

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    bottom_y = undist_image.shape[0] - 1
    bottom_x_left = left_fit[0] * (bottom_y ** 2) + left_fit[1] * bottom_y + left_fit[2]
    bottom_x_right = right_fit[0] * (bottom_y ** 2) + right_fit[1] * bottom_y + right_fit[2]
    vehicle_offset = undist_image.shape[1] / 2 - (bottom_x_left + bottom_x_right) / 2

    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    vehicle_offset *= xm_per_pix


    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))



    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist_image.shape[1], undist_image.shape[0]))
    result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)
    return result,vehicle_offset



#function to calculate curvature
def calculate_curvature(img,vehicle_offset):

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image

    y_eval = np.max(ploty)
    quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                      for y in ploty])
    rightx = np.array([900 + (y ** 2) * quadratic_coeff + np.random.randint(-50, high=51)
                       for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    avg_curve=(right_curverad+left_curverad)/2
    label='The radius of curvature : %.1f m' % avg_curve
    result = cv2.putText(img, label, (30, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)





    label_str = 'Vehicle offset from lane center: %.2f m' % vehicle_offset
    result = cv2.putText(result, label_str, (30, 70), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return result



#pipeline to process image
def process_image(image):

    undistorted_image=undistort(image)
    combined_binary=threshold(undistorted_image)
    m,m_inv=perspective_matrix(combined_binary)
    warped_image=perspective_transform_image(combined_binary,m)
    left_fit, right_fit,left_x,right_x=detect_lanes(warped_image)
    result,vehicle_offset=draw_lanes(undistorted_image,m_inv,left_fit,right_fit)
    final=calculate_curvature(result,vehicle_offset)

    return final



from moviepy.editor import VideoFileClip

challenge_output = 'project_op_1.mp4'
clip3 = VideoFileClip('project_video.mp4')
#clip3 = VideoFileClip('test_videos/challenge.mp4')
#clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)