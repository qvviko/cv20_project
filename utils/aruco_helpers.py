import aruco
import cv2
import numpy as np


# Helper function
# Given the marker map and location of the center of the box, computers the 3D position of the center of the box
def get_center_box_pos(mmap, center_location):
    rt_matrix = mmap.getRTMatrix()[:-1, :]  # Q|t matrix
    vector_3D = rt_matrix @ np.append(center_location, 1)  # [Q|t] * homogeneous_center_box
    return vector_3D


# Helper function
# Given the location of the 3D vector and camera matrix, calculates the location in the image
# Returns x and y
def get_2d_from_3d(vector_3d, camera_matrix):
    d2 = camera_matrix @ vector_3d  # Camera matrix * vector_3d
    return int(d2[0] / d2[-1]), int(d2[1] / d2[-1])


# Init a new marker detector and two pose trackers
def get_new_MM_posers(camera_parameters, mmap_low, mmap_high):
    low_pose = aruco.MarkerMapPoseTracker()
    high_pose = aruco.MarkerMapPoseTracker()

    low_pose.setParams(camera_parameters, mmap_low)
    high_pose.setParams(camera_parameters, mmap_high)
    return aruco.MarkerDetector(), low_pose, high_pose


# Helper function to get highs and lows for marker
def get_heights_width_markers(lower_marker, upper_marker):
    min_h, max_w = np.array(lower_marker).mean(axis=0)
    max_h, min_w = np.array(upper_marker).mean(axis=0)
    return int(min_h), int(max_h), int(min_w), int(max_w)


# Helper function to draw the line for given bounds of the boxes
def draw_line(image, min_h, min_w, max_h, max_w, thickness=1, color=(255, 0, 0)):
    to_return = image.copy()
    to_return = cv2.line(to_return, (0, max_w - min_w), (max_h - min_h, 0), color, thickness)
    return to_return


# Helper function to highlight the wire in the image
def search_for_line(image):
    to_return = image.copy()
    to_return = cv2.cvtColor(to_return, cv2.COLOR_RGB2GRAY)
    to_return = cv2.GaussianBlur(to_return, (7, 7), 0)  # Remove inconsistencies

    # Find a line
    to_return = cv2.adaptiveThreshold(to_return, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 12)

    # Remove some noise
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    to_return = cv2.erode(to_return, kernel_erosion)
    to_return = cv2.dilate(to_return, kernel_dilation)

    return to_return


# Helper class to store the amount of pixels in a row
class Ray:
    def __init__(self):
        self.starting_pos = None
        self.width = 0

    def add_point(self, point):
        # If there is no starting point, it will become one
        if self.starting_pos is None:
            self.starting_pos = point
        self.width += 1

    def get_middle(self):
        # Get the middle of the line
        x, y = self.starting_pos
        return y, x + int(self.width / 2)


# Helper function to transform points from 3D to 2D given camera parameters, t vector and r vector
def get_2d_points(camera_params, tvec, rvec, distortion=None):
    if distortion is None:
        distortion = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    point_3d = np.array([0, 0, 0], dtype=np.float32)
    point = cv2.projectPoints(np.array([[point_3d]]), rvec, tvec, camera_params, distortion)
    x, y = tuple(point[0][0][0])
    return int(x), int(y)


# Helper function
# Given the image and initial coordinate of x and y on the image
# try to find y that will be in the middle of the line
def get_middle_y(img, x, y):
    dynamic_y = y
    # Firstly, measure the number of white pixels from the top
    while img[dynamic_y, x] == 255:
        dynamic_y += 1
    height_below = dynamic_y - y

    dynamic_y = y
    # Then measure number of nwhite pixels on the bottom
    while img[dynamic_y, x] == 255:
        dynamic_y -= 1
    height_above = y - dynamic_y

    # Calculate mean
    mean = int((height_below + height_above) / 2)

    # Adjust the y location, so that the number of white pixels at the top and at the bottom is the same
    return y - (height_above - mean)


# Collect points along the line, given the location of the middle of the line
def find_points(img, middle):
    x, y = middle
    h, w = img.shape
    xs = [x]
    ys = [y]

    # Go to through right direction until there is no cable
    tmp_x = x + 1
    tmp_y = y
    while w > tmp_x and h > tmp_y and img[tmp_y, tmp_x] == 255:
        tmp_y = get_middle_y(img, tmp_x, tmp_y)  # Adjust y to be in the middle
        xs.append(tmp_x)
        ys.append(tmp_y)
        tmp_x += 1

    # Go throught the left direction until there is no cable
    tmp_x = x - 1
    tmp_y = y
    while w > tmp_x and h > tmp_y and img[tmp_y, tmp_x] == 255:
        tmp_y = get_middle_y(img, tmp_x, tmp_y)  # Adjust y to be in the middle
        xs.append(tmp_x)
        ys.append(tmp_y)
        tmp_x -= 1

    return xs, ys
