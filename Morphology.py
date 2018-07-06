import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import networkx as nx
# from scipy import weave
from typing import Tuple, List, Dict

from scipy.ndimage import rotate, shift

import s4395897_ELEC4630_module.DiscreteLogic as ds



def add_lines(frame, lines, color=(255,0,0), thickness=2):
    for line in lines:
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), color, thickness)


def add_points(frame, points, color=(255,0,0), radius=3):
    for point in points:
        cv2.circle(frame, (point[0], point[1]), radius, color, -1)


"""
Applied the lower and upper thresholds to the given rbg image
in HSV space. This returns the resulting binary image.
@param: hue -> 1x2 array of hue between 0 - 1
@param: saturation -> 1x2 array of saturation between 0-1
@param: ...
"""
def hsv_threshold(image, hue, saturation, value, maintainValues=False):
    # This is a BGR image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if not isinstance(hue, np.ndarray):
        hue = np.array(hue)
    if not isinstance(saturation, np.ndarray):
        saturation = np.array(saturation)
    if not isinstance(value, np.ndarray):
        value = np.array(value)

    # Hue
    changeHue = False
    if hue[0] >= hue[1]:
        # Reverse the array
        print("Reverse hue")
        changeHue = True
        hue = hue[::-1]

    hue *= 180

    # Saturation
    saturation *= 255

    # intensity
    intensity = value * 255

    # Filter
    lower = np.array([hue[0], saturation[0], intensity[0]])
    upper = np.array([hue[1], saturation[1], intensity[1]])

    if maintainValues:
        for index, cellValue in np.ndenumerate(hsv):
            row = hsv[index[0], index[1]]

            if row[0] < hue[0] or row[0] > hue[1]:
                hsv[index[0], index[1]] = np.array([0.0, 0.0, 0.0], np.uint8)
                break

            if row[1] < saturation[0] or row[1] > saturation[1]:
                hsv[index[0], index[1]] = np.array([0.0, 0.0, 0.0], np.uint8)
                break

            if row[2] < intensity[0] or row[2] > intensity[1]:
                hsv[index[0], index[1]] = np.array([0.0, 0.0, 0.0], np.uint8)
                break

        return hsv

    # Handle the case in which we want hues that pass the start point
    if not changeHue:
        return cv2.inRange(hsv, lower, upper)
    else:
        print("Special Hue handling...")
        zero = np.array([0, 1.0]) * 180
        newLower = np.array([0.0, saturation[0], intensity[0]])
        newUpper = np.array([180.0, saturation[1], intensity[1]])

        noHue = cv2.inRange(hsv, newLower, newUpper)
        hueFilter = cv2.inRange(hsv, lower, upper)

        return noHue - hueFilter


"""
Applies a dilation to the given image using the provided kernel. 
"""
def dilate_image(image, kernelDimensions=(9,9), iters=1, customKernal=None):
    imageCopy = image.copy() # Copy the initial image

    if customKernal is None:
        # Create the kernel for use. Note: dimensions are in the form (y, x)
        kernel = np.ones(kernelDimensions, np.uint8)
    else:
        kernel = customKernal

    return cv2.dilate(imageCopy, kernel, iterations = iters)


"""
Applies an erosion to the given image using the provided kernel. 
"""
def erode_image(image, kernelDimensions=(9, 9), iters=1, kernel=None):
    imageCopy = image.copy() # Copy the initial image

    # Create the kernel for use. Note: dimensions are in the form (y, x)
    if kernel is None:
        kernel = np.ones(kernelDimensions, np.uint8)
    return cv2.erode(imageCopy, kernel, iterations = iters)


"""
Applies an opening to the given image using the provided kernel. 

Inputs:
image               -> The image to process.
kernelDimensions    -> The dimensions of a rectangle to use.
iters               -> The number of iterations to undertaken.
customKernal        -> A non-standard kernel to use if provided.
"""
def open_image(image: np.ndarray, kernelDimensions=(3, 3), iters=1, customKernel=None):
    imageCopy = image.copy() # Copy the image.

    # Create the kernel
    if customKernel is None:
        kernel = np.ones(kernelDimensions, np.uint8)
    else:
        kernel = customKernel

    # Erode then dilate the image using the same kernel
    imageCopy = cv2.erode(imageCopy, kernel, iterations = iters)
    imageCopy = cv2.dilate(imageCopy, kernel, iterations = iters)

    return imageCopy


"""
Applies a closing to the given image using the provided kernel. 

Inputs:
image               -> The image to process.
kernelDimensions    -> The dimensions of a rectangle to use.
iters               -> The number of iterations to undertaken.
customKernal        -> A non-standard kernel to use if provided.
"""
def close_image(image, kernelDimensions=(9, 9), iters=3, customKernel=None):
    imageCopy = image.copy() # Copy the initial image.

    # Create the kernel
    kernel = np.ones(kernelDimensions, np.uint8)
    if customKernel is not None:
        kernel = customKernel

    # Dilate then erode using the same kernel
    imageCopy = cv2.dilate(imageCopy, kernel, iterations = iters)
    imageCopy = cv2.erode(imageCopy, kernel, iterations = iters)

    return imageCopy


"""
Finds the boundary of the objects within the given image.
"""
def get_boundary(image, kernelDimensions = (4,3)):
    # A - ( A erode by B)
    return cv2.bitwise_xor(image, erode_image(image, kernelDimensions))


"""
Finds the skeleton of the given image using the provided dimensioned kernel. The smaller
the kernel the faster the algorithm runs.

image       -> A binary mage to find the skeletons of each object in.
dim         -> The dimensions of the kernel to apply.
"""
def get_skeleton(image, dim):
    skel = np.zeros(image.shape, dtype=np.uint8)
    size = np.size(image)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, dim)
    finished = False
    img = image

    # plt.subplot(2,2,1)
    # plt.imshow(img)

    while not finished:
        eroded = cv2.erode(img, element)
        # plt.subplot(2,2,2)
        # plt.imshow(eroded)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        # plt.subplot(2,2,3)
        # plt.imshow(temp)
        skel = cv2.bitwise_or(skel, temp)
        # plt.subplot(2,2,4)
        # plt.imshow(skel)
        # plt.show()
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        print(zeros, size)
        if zeros == size:
            finished = True

    return skel


import scipy as sp
import scipy.ndimage

def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array)
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array



"""
Given a binary image, labels each connected object with a label. These labels are an integer value from 1.
"""
def label_connected_objects(image):
    ret, labels = cv2.connectedComponents(image)

    return labels.astype(np.uint8)


"""
Removes all binary objects that arn't the largest.
"""
def remove_smaller_then(image, minSize):
    ret, labels = cv2.connectedComponents(image)
    unique, counts = np.unique(labels, return_counts=True)
    counts = dict(zip(unique, counts))

    newImage = np.zeros(image.shape, np.uint8)

    toAdd = []
    for label, num in counts.items():
        if num > minSize:
            toAdd.append(label)
    toAdd.remove(0)

    # print("Labels:", len(labels))
    # for label in toRemove:
    #     labels[labels == label] = 0

    # print(toAdd)
    for label in toAdd:
        newImage[labels == label] = 255

    return newImage


"""
Removes all binary objects that arn't the largest.
"""
def remove_larger_then(image, maxSize):
    ret, labels = cv2.connectedComponents(image)
    unique, counts = np.unique(labels, return_counts=True)
    counts = dict(zip(unique, counts))

    newImage = np.zeros(image.shape, np.uint8)

    toAdd = []
    for label, num in counts.items():
        if num < maxSize:
            toAdd.append(label)

    if 0 in toAdd:
        toAdd.remove(0)

    # print("Labels:", len(labels))
    # for label in toRemove:
    #     labels[labels == label] = 0

    # print(toAdd)
    for label in toAdd:
        newImage[labels == label] = 255

    return newImage


"""
Removes all binary components that touch the edge of the image.
"""
def remove_edge_connected(image):
    ret, labels = cv2.connectedComponents(image)

    # Left most edge
    toRemove = np.where(labels[0, :] > 0)
    for label in toRemove:
        for val in np.unique(labels[0, label]):
            labels[labels == val] = 0

    # Left most edge
    toRemove = np.where(labels[labels.shape[0] - 1, :] > 0)

    for label in toRemove:
        for val in np.unique(labels[labels.shape[0] - 1, label]):
            labels[labels == val] = 0

    # Left most edge
    toRemove = np.where(labels[:, 0] > 0)

    for label in toRemove:
        for val in np.unique(labels[label, 0]):
            labels[labels == val] = 0

    # Left most edge
    toRemove = np.where(labels[:, labels.shape[1] - 1] > 0)

    for label in toRemove:
        for val in np.unique(labels[label, labels.shape[1] - 1]):
            labels[labels == val] = 0

    labels[labels > 0] = 255

    return labels.astype(np.uint8)


def keep_largest_objects(image, numObjects: int = 1):
    ret, labels = cv2.connectedComponents(image)
    unique, counts = np.unique(labels, return_counts=True)
    counts = dict(zip(unique, counts))

    newImage = np.zeros(image.shape, np.uint8)

    sortedLabels = []
    for label, num in counts.items():
        if label == 0:
            continue
        sortedLabels.append((num, label))

    sortedLabels.sort()

    for label in sortedLabels[-(numObjects):]:
        newImage[labels == label[1]] = 255

    return newImage


def remove_largest_objects(image, numObjects, skip=0):
    # Label connected components
    ret, labels = cv2.connectedComponents(image)
    unique, counts = np.unique(labels, return_counts=True)
    counts = dict(zip(unique, counts))

    # Sort based on the number of elements
    sortedLabels = []
    for label, num in counts.items():
        if label == 0:
            continue
        sortedLabels.append((num, label))
    sortedLabels.sort()

    # Set largest n objects to 0
    for label in sortedLabels[-(numObjects) - (skip): len(sortedLabels)-(skip)]:
        labels[labels == label[1]] = 0

    # remove all labelling information
    labels[labels > 0] = 255
    labels = labels.astype(np.uint8)

    return labels


cropY = (0, 910)
cropX = (0, 2100)

def crop_frame(frame, cropX, cropY):

    return frame[cropY[0]:cropY[1], cropX[0]:cropX[1]]
    #frame[int(yLen * cropY[0]):int(yLen * cropY[1]), int(xLen * cropX[0]): int(xLen * cropX[1])]


def uncrop_point(point):
    return point[0] + cropX[0], point[1] + cropY[0]

    #return #(int(point[0] + xLen * cropX[0]), int(point[1] + yLen * cropY[0]))


def find_circles(image, min_between_circles = 1, edge_detect_threshold = 40, accum_threshold = 20, minRadius = 20, maxRadius = 0):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, min_between_circles, param1 = edge_detect_threshold,
                               param2 = accum_threshold, minRadius = minRadius, maxRadius = maxRadius)

    if circles is None:
        print("No circles")
        return []

    circles = np.uint16(np.around(circles))

    return [c for c in circles][0]


# CHANGE HERE !!!! <----------------------
def rot(image, xy, angle):
    im_rot = rotate(image,angle)
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot, new+rot_center


def trans(image, x, y):
    yLen, xLen = image.shape

    translated = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for (i, j), value in np.ndenumerate(image):
        if i + y < yLen and j + x < xLen and i + y >=0 and j + x >= 0:
            translated[i + y, j + x] = value

    return translated


"""
Given a closed outline of any shapes, fill in the edges to make a solid shape.
"""
def fill_closed_space_binary(space, edgeIntensity):
    space = space.copy()

    # Sweep across
    for y in range(space.shape[0]):
        # sweep from left to right and mark all points after 128 as 2
        foundPoint = False
        for x in range(space.shape[1]):
            if foundPoint == False and space[y, x] == edgeIntensity:
                foundPoint = True

            if foundPoint == True and space[y, x] != edgeIntensity:
                space[y, x] = 2

        # sweep from right to left and mark all 2's after a 255 as 3's
        foundPoint = False
        for x in range(space.shape[1] -1, 0 - 1, -1):
            if foundPoint == False and space[y, x] == edgeIntensity:
                foundPoint = True

            if foundPoint == True and space[y, x] == 2:
                space[y, x] = 3

    # Remove all features
    space[space == 3] = 255
    space[space == 2] = 0

    return space


def find_bounding_box(image):
    pass


def bin_to_graph(binary):
    ret, labels = cv2.connectedComponents(image)
    unique, counts = np.unique(labels, return_counts=True)

    graph = nx.Graph()

    for label in labels:
        for index, value in np.ndenumerate(labels):
            if value == label:
                graph.add_node(index)
                # Check surrounding 8 pixels for other labels

                graph.add_edge()


import bisect
from numba import jit

"""
Credit to:
https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python
"""
@jit
def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 256): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst.astype(np.uint8)


"""
Given a greyscale image, adjusts the intensity so that it is evenly distribution over the entire 256 bit range.
Inputs: image   -> A greyscale opencv image (2d np.ndarray)
Ouputs: image   -> A greyscale opencv image (2d np.ndarray) with the intensity distribution over the entire range.
"""
def adjust_intensity(image: np.ndarray) -> np.ndarray:
    if len(image.shape) != 2:
        raise ValueError("Image should be greyscale (2d numpy array)")

    # Find the minimum and maximum values
    pass





def neighbours_vec(image):
    return image[2:,1:-1], image[2:,2:], image[1:-1,2:], image[:-2,2:], image[:-2,1:-1],     image[:-2,:-2], image[1:-1,:-2], image[2:,:-2]

def transitions_vec(P2, P3, P4, P5, P6, P7, P8, P9):
    return ((P3-P2) > 0).astype(int) + ((P4-P3) > 0).astype(int) + \
    ((P5-P4) > 0).astype(int) + ((P6-P5) > 0).astype(int) + \
    ((P7-P6) > 0).astype(int) + ((P8-P7) > 0).astype(int) + \
    ((P9-P8) > 0).astype(int) + ((P2-P9) > 0).astype(int)

def zhangSuen_vec(image, iterations):
    for iter in range (1, iterations):
        # step 1
        P2,P3,P4,P5,P6,P7,P8,P9 = neighbours_vec(image)
        condition0 = image[1:-1,1:-1]
        condition4 = P4*P6*P8
        condition3 = P2*P4*P6
        condition2 = transitions_vec(P2, P3, P4, P5, P6, P7, P8, P9) == 1
        condition1 = (2 <= P2+P3+P4+P5+P6+P7+P8+P9) * (P2+P3+P4+P5+P6+P7+P8+P9 <= 6)
        cond = (condition0 == 1) * (condition4 == 0) * (condition3 == 0) * (condition2 == 1) * (condition1 == 1)
        changing1 = np.where(cond == 1)
        image[changing1[0]+1,changing1[1]+1] = 0
        # step 2
        P2,P3,P4,P5,P6,P7,P8,P9 = neighbours_vec(image)
        condition0 = image[1:-1,1:-1]
        condition4 = P2*P6*P8
        condition3 = P2*P4*P8
        condition2 = transitions_vec(P2, P3, P4, P5, P6, P7, P8, P9) == 1
        condition1 = (2 <= P2+P3+P4+P5+P6+P7+P8+P9) * (P2+P3+P4+P5+P6+P7+P8+P9 <= 6)
        cond = (condition0 == 1) * (condition4 == 0) * (condition3 == 0) * (condition2 == 1) * (condition1 == 1)
        changing2 = np.where(cond == 1)
        image[changing2[0]+1,changing2[1]+1] = 0
    return image




def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1
                    P2 * P4 * P6 == 0  and    # Condition 3
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


def _thinningIteration(im, iter):
	I, M = im, np.zeros(im.shape, np.uint8)
	expr = """
	for (int i = 1; i < NI[0]-1; i++) {
		for (int j = 1; j < NI[1]-1; j++) {
			int p2 = I2(i-1, j);
			int p3 = I2(i-1, j+1);
			int p4 = I2(i, j+1);
			int p5 = I2(i+1, j+1);
			int p6 = I2(i+1, j);
			int p7 = I2(i+1, j-1);
			int p8 = I2(i, j-1);
			int p9 = I2(i-1, j-1);
			int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
			         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
			         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
			         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
			if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
				M2(i,j) = 1;
			}
		}
	} 
	"""

	weave.inline(expr, ["I", "iter", "M"])
	return (I & ~M)


def thinning(src):
	dst = src.copy() / 255
	prev = np.zeros(src.shape[:2], np.uint8)
	diff = None

	while True:
		dst = _thinningIteration(dst, 0)
		dst = _thinningIteration(dst, 1)
		diff = np.absolute(dst - prev)
		prev = dst.copy()
		if np.sum(diff) == 0:
			break

	return dst * 255


"""
Given a image with a border, which can be connected to inner components. This border is removed maintaining as much
of the inner object as possible.
"""
def remove_border(image: np.ndarray, kernelEstimate: Tuple[int, int] =(11,11)):
    # Remove small inner objects
    openned = open_image(image, kernelEstimate)

    # Subtract the border
    result = cv2.subtract(image, openned)

    return result


"""
Given lines that form a partial rectangle, complete the rectangle by extending the sides of the lines until they all intersect.
Edge case: Behaviour for parralel lines is undefined.

Inputs:
Lines       -> lines: A list of 4 lines that form a rectangle.

Returns: 4 lines that form a rectangle.
"""
def complete_rectange(lines):
    # Move to Hough coordinates
    lines = convert_lines_to_hough_coords(lines)

    # Find the intersections of the lines
    # Find the 2 lines with similar gradients
    pair1 = []

    pair1.append(lines[0])
    pair1.append(lines[1])
    for line in lines[2:]:
        if abs(pair1[1][0] - pair1[0][0]) > abs(line[0] - pair1[0][0]):
            pair1[1] = line

    pair2 = []
    for line in lines:
        if line not in pair1:
            pair2.append(line)

    # Find the intersecting corners
    corners = []
    for i, line1 in enumerate(pair1):
        for j, line2 in enumerate(pair2):
            # Find the intersection of the lines in each pair
            corners.append(find_hough_space_line_intersections(line1, line2))

    # Error checking
    for c in corners:
        if np.isnan(c):
            raise ValueError()

    # join corners into lines
    newLines = []
    newLines.append(np.array([*corners[0], *corners[1]]).astype(np.int))
    newLines.append(np.array([*corners[2], *corners[3]]).astype(np.int))
    newLines.append(np.array([*corners[0], *corners[2]]).astype(np.int))
    newLines.append(np.array([*corners[1], *corners[3]]).astype(np.int))

    return newLines




    extendedLines = []

    # Add length to all lines
    for (x1, y1, x2, y2) in lines:
        # Find the length of the line
        length = ds.line_length((x1, y1, x2, y2))
        extendedLines.append(extend_line(x1, y1, x2, y2, int(length * 0.5)))


    # Find the 2 lines with similar gradients
    pair1 = []

    pair1.append(linesHoughCoords[0])
    pair1.append(linesHoughCoords[1])
    for line in linesHoughCoords[2:]:
        if abs(pair1[1][0] - pair1[0][0]) < abs(line[0] - pair1[0][0]):
            pair1[1] = line

    pair2 = []
    for line in linesHoughCoords:
        if line not in pair1:
            pair2.append(line)

    # check if the rectangle is complete
    corners = []
    for i, line1 in enumerate(pair1):
        for j, line2 in enumerate(pair2):
            # Find the intersection of the lines in each pair
            corners.append(find_hough_space_line_intersections(line1, line2))

    print(corners)

    # Trim the lines
    pass



def convert_lines_to_hough_coords(lines):
    # move to hough coordinates to prevent inifinite gradients
    # r = x cos(\theta) + y sin(\theta)
    # x_1 = X_1, y_1 = Y_1
    # x_2 = X_2, y_2 = Y_2
    # r = X_1 * cos( \theta ) + Y_1 sin( \theta )
    #
    # r = X_2 * cos( \theta ) + Y_2 sin( \theta )
    #
    # Equate:
    # X_1 * cos( \theta ) + Y_1 sin( \theta ) = X_2 * cos( \theta ) + Y_2 sin( \theta )
    # X_1 + Y_1 tan( \theta ) = X_2 + Y_2 tan( \theta )
    # (Y_1 - Y_2 )tan( \theta ) = X_2 - X_1
    # tan( \theta ) = (X_2 - X_1) / (Y_1 - Y_2)
    #
    # Solve:
    # \theta = arctan( (X_2 - X_1) / (Y_1 - Y_2) )
    # r = X_1 * cos( \theta ) + Y_1 sin( \theta )

    # Convert the lines to be in the form: \theta, \rho
    linesHoughCoords = []
    for (x1, y1, x2, y2) in lines:
        if y1 - y2 == 0:
            y1 += 1

        theta = math.atan(((x2 - x1) / (y1 - y2)))
        rho = x1 * math.cos(theta) + y1 * math.sin(theta)

        linesHoughCoords.append([theta, rho])

    return linesHoughCoords


def find_hough_space_line_intersections(line1, line2):
    # r_1 = x * cos( \theta_1 ) + y sin( \theta_1 )
    # x = (r1 - y sin (\theta_1) ) / cos( \theta_1)
    #
    # r_2 = x * cos( \theta_2 ) + y sin( \theta_2 )
    # x = (r2 - y sin (\theta_2) ) / cos( \theta_2)
    #
    # (r1 - y sin (\theta_1) ) / cos( \theta_1) = (r2 - y sin (\theta_2) ) / cos( \theta_2)
    # r1 cos(\theta_2) - y sin(theta1) cos(theta2) = r2 cos(\theta_1) - y sin(theta2) cos(theta1)
    # - y sin(theta1) cos(theta2) + y sin(theta2) cos(theta1) = r2 cos(\theta_1) - r1 cos(\theta_2)
    #
    # y = (r2 cos(\theta_1) - r1 cos(\theta_2)) / ( sin(theta2) cos(theta1) - sin(theta1) cos(theta2) )
    # x = (r1 - y sin (\theta_1) ) / cos( \theta_1)

    theta1, rho1 = line1
    theta2, rho2 = line2

    # Correct for rounding errors
    if  math.sin(theta2) * math.cos(theta1) - math.sin(theta1) * math.cos(theta2) == 0 or math.cos(theta1) == 0:
        return (np.nan, np.nan)

    y = (rho2 * math.cos(theta1) - rho1 * math.cos(theta2)) / (math.sin(theta2) * math.cos(theta1) - math.sin(theta1) * math.cos(theta2))
    x = (rho1 - y * math.sin(theta1)) / math.cos(theta1)

    return (x, y)



"""
Given a line, extend it by the given amount.

Inputs: 
x1, y1, x2, y2  -> The line to be extended.
increaseBy      -> The amount to increase the line by as a percentage increase.
"""
def extend_line(x1: int, y1: int, x2: int, y2: int, increaseBy: int):
    # TODO: Please clean up this method

    v = np.array((x2 - x1, y2 - y1))
    vDash = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    vHat = v / vDash  # Rise, Run

    if isinstance(increaseBy, float):
        AExtended = np.array((x1, y1)) - vDash * increaseBy * vHat
        BExtended = np.array((x2, y2)) + vDash * increaseBy * vHat
    else:
        AExtended = np.array((x1, y1)) - increaseBy * vHat
        BExtended = np.array((x2, y2)) + increaseBy * vHat

    # Ensure that the line does not extend to below (0,0)
    if AExtended[0] < 0:
        if (BExtended[0] - AExtended[0]) == 0:
            m = 100000
        else:
            m = (BExtended[1] - AExtended[1]) / (BExtended[0] - AExtended[0])
        # c = y - mx
        c = AExtended[1] - m * AExtended[0]

        AExtended[0] = 0
        AExtended[1] = c


    if AExtended[1] < 0:
        if (BExtended[0] - AExtended[0]) == 0:
            m = 100000
        else:
            m = (BExtended[1] - AExtended[1]) / (BExtended[0] - AExtended[0])

        # c = y - mx
        c = AExtended[1] - m * AExtended[0]

        # x = -c / m
        AExtended[0] = - c / m
        AExtended[1] = 0

    return int(round(AExtended[0])), int(round(AExtended[1])), int(round(BExtended[0])), int(round(BExtended[1]))



# def intersectLines( pt1, pt2, ptA, ptB ):
def intersectLines(line1, line2):
    pt1, pt2 = (line1[0], line1[1]), (line1[2], line1[3])
    ptA, ptB = (line2[0], line2[1]), (line2[2], line2[3])

    # TODO: CHANGE THIS
    """
    ALL credit to: https://www.cs.hmc.edu/ACM/lectures/intersections.html
    this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)"""

    tol = 0.00001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    deltax1 = x2 - x1;  deltay1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB;
    dx = xB - x;  dy = yB - y;

    determinant = (-deltax1 * dy + deltay1 * dx)

    if math.fabs(determinant) < tol: return None

    # now, the determinant should be OK
    inverserDeterminate = 1.0/determinant

    # find the scalar amount along the "self" segment
    r = inverserDeterminate * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = inverserDeterminate * (-deltay1 * (x-x1) + deltax1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*deltax1 + x + s*dx)/2.0
    yi = (y1 + r*deltay1 + y + s*dy)/2.0

    # Ensure the point lines on both lines
    if not isBetween(pt1, pt2, (xi, yi)) or not isBetween(ptA, ptB, (xi, yi)):
        return None

    return (int(round(xi)), int(round(yi)))# (xi, yi, 1, r, s )





if __name__ == "__main__":
    shape = np.array([[1, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0]], dtype=np.uint8)

    print(shape)

    print(trans(shape, 3, 0))

    print(trans(shape, 2, 2))

    print(trans(shape, -1, 0))

    print(trans(shape, 0, -2))
