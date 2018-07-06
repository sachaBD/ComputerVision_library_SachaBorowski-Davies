from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from OpenGL.GL import *
from OpenGL.GLU import *



def create_subplot(numImages):
    # Use a square
    sideLen = int(np.ceil(np.sqrt(numImages)))

    fig, ax = plt.subplots(sideLen, sideLen, squeeze=False)

    return fig, ax

def get_dimensions(index, ax):
    x, y = ax.shape
    return index // x, index % x


def plot_mask(original, mask, colour=(0,255,0)):
    original[mask > 0] = colour

    return original


def add_lines(frame, lines, color=(255,0,0), thickness=2):
    for line in lines:
        try:
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), color, 2)
        except:
            line = line[0]
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), color, 2)

def add_points(frame, points, color=(255,0,0), radius=3):
    for point in points:
        cv2.circle(frame, (point[0], point[1]), radius, color, -1)

def add_circles(frame, points, color=(255,0,0), thickness = 3):
    for point in points:
        cv2.circle(frame, (point[0], point[1]), point[2], color, thickness)


"""
Adds a line between each point onto the frame.

Inputs:
frame       -> The frame to draw on.
Points      -> A list of tuples of points to join. In order (x, y)
closed      -> If the shape is closed or not.
"""
def add_connected_points(frame: np.ndarray, points: List[Tuple[int, int]], colour=(255,0,0), closed=True, thickness = 1):
    for j in range(len(points) - 1):
        x1, y1 = points[j]
        x2, y2 = points[j + 1]
        cv2.line(frame, (x1, y1), (x2, y2), colour, thickness=1)  # , lineType=8)

    # Connect the start and end
    if closed:
        x1, y1 = points[0]
        x2, y2 = points[-1]
        cv2.line(frame, (x1, y1), (x2, y2), colour, thickness=1)

    return frame


"""
Loads all files from the given folder with the .extension as images.
Inputs:
folderName      -> String of the foldername.
extensions      -> The extension to load with the .
"""
def load_from_folder(folderName, extension, greyScale=False):
    frames = []

    for file in os.listdir(folderName):
        if file[-4:] == "." + extension:
            print(folderName + "\\" + file)

            if greyScale == False:
                frames.append(cv2.imread(folderName + "\\" + file, cv2.IMREAD_COLOR))
            else:
                frames.append(cv2.imread(folderName + "\\" + file, cv2.IMREAD_GRAYSCALE))

    return frames

"""
Overlays the given image over the background image.
Inputs:
background      -> The image to overlay over. This must have 3 colour channel.
overlay         -> The image to overlay, can be single or multi channel.
"""
def add_overlay(background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    if len(background.shape) == 2:
        # Create a mask for the overlay
        if len(overlay.shape) == 3:
            mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        else:
            mask = overlay

        mask[mask > 0] = 255
        mask = cv2.bitwise_not(mask)

        # Mask the original image
        final = cv2.bitwise_and(background, background, mask=mask)

        # Add the overlay to the original image
        print("Background", background.shape, background.dtype, "overlay", overlay.shape, overlay.dtype)
        final = cv2.bitwise_or(background, overlay)

        return final

    else:
        ogOverlay = overlay.copy()
        # Create a mask for the overlay
        if len(overlay.shape) == 3:
            mask = cv2.cvtColor(overlay.copy(), cv2.COLOR_BGR2GRAY)
        else:
            mask = overlay.copy()

        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            overlay[:, :, 0] = 0
            overlay[:, :, 2] = 0
            overlay = overlay * 3

        # t = ogOverlay.copy()
        # colT = cv2.cvtColor(t, cv2.COLOR_GRAY2RGB)
        # plt.subplot(2,1,1)
        # plt.imshow(t)
        # plt.subplot(2,1,2)
        # plt.imshow(overlay)
        # plt.show()

        mask[mask > 0] = 255
        mask = cv2.bitwise_not(mask)

        # Mask the original image
        final = cv2.bitwise_and(background, background, mask=mask)

        # Add the overlay to the original image
        print("Background", background.shape, background.dtype, "overlay", overlay.shape, overlay.dtype)
        final = cv2.bitwise_or(background, overlay)

        # plt.imshow(overlay)
        # plt.show()
        #
        # plt.imshow(final)
        # plt.show()

        return final


"""
Create a 3d cube of the given size.
Source: https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib
"""
def cuboid_data(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

"""
Plots a cube at the given position with the size and colour specified.
Source: https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib
"""
def plotCubeAt(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for i, (p,s,c) in enumerate(zip(positions,sizes,colors)):
        g.append( cuboid_data(p, size=s) )

        if i % 10000 == 0:
            print(i)

    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors,6, axis=0), **kwargs)



def create_voxel_space(x, y, z):
    space = np.indices((x, y, z))
    return space[0]



"""
Given a 3d array of points, plots them as voxels.
"""
def plot_voxels(voxels):
    # prepare some coordinates
    # space = np.indices((8, 8, 8))
    # space = space[0]

    # draw cuboids in the top left and bottom right corners, and a link between them
    #cube1 = (x < 3) & (y < 3) & (z < 3)
    #cube2 = (x >= 5) & (y >= 5) & (z >= 5)
    #link = abs(x - y) + abs(y - z) + abs(z - x) <= 2
    #myCube = ((x > 4) & (x < 7)) & ((y > 4) & (y < 7)) & ((z > 5) & (z < 7))
    # print(myCube.shape)
    # myCube = space
    # print(myCube.shape)

    # combine the objects into a single boolean array
    #voxels = cube1 | cube2 | link
    # voxels = myCube

    # set the colors of each object
    # colors = np.empty(voxels.shape, dtype=object)
    # colors[link] = 'red'
    # colors[cube1] = 'blue'
    # colors[cube2] = 'green'
    # colors[myCube] = 'green'

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.voxels(voxels, edgecolor='k')


    plt.show()



def openGLPlot():
    pass




if __name__ == "__main__":
    v = create_voxel_space(100, 100, 100)
    print("Created voxels")
    plot_voxels(v)