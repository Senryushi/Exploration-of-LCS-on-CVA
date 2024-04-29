import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


### SUMMARY

# This program gets the optical flow field for a given video. The output is
# given in a window. You can press "Esc" to close the window and "s" to
# save the frame you are on as an image file.

# Get the search path for sample data,
# and open the video capture device for the video file
cv.samples.addSamplesDataSearchPath("Folder_path")
cap = cv.VideoCapture(cv.samples.findFile("File_path"))

# Total number of frames in the video
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# Read the first frame and convert it to greyscale
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# Create a HSV image with all pixels set to maximum saturation (255)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# Calculate optical flow for each frame
for i in range(length - 1):
    ret, next = cap.read()
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 5, 10, 3, 5, 1, 0)

    # Convert flow to magnitude and angle
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Create a HSV image from the flow data
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # Convert HSV to BGR for display
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Display the flow image
    cv.imshow("frame2", bgr)

    # Exit on 'Esc' key press and Save images on 's' key press
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break
    elif k == ord("s"):
        cv.imwrite("opticalfb.png", flow)
        cv.imwrite("opticalhsv.png", bgr)

    # Update the previous frame
    prvs = next

# Close all OpenCV windows
cv.destroyAllWindows()
