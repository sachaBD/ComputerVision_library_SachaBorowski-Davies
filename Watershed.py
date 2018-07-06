import numpy as np
import cv2
import matplotlib.pyplot as plt
from bintrees import AVLTree



def watershed(image, markers):
    # Only check directly straight connections
    adjacent = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])

    # Constants
    boundary, unset, unfound, maxMarker = 0, 1, -1, 258

    # compute the gradient image
    gradient = np.abs(cv2.Laplacian(image, cv2.CV_64F))

    # Initialise
    for index, value in np.ndenumerate(markers):
        if value > 0:
            gradient[index[0], index[1]] = value + 255

    gradient = gradient + 2

    # Store which points need checking
    toCheck = AVLTree()
    for i in range(1,  gradient.shape[0] - 2):
        for j in range(1, gradient.shape[1] - 2):
            if gradient[i, j] < maxMarker:
                toCheck.insert((i, j), gradient[i, j])


    for i in range(2, 257):
        print(i, len(toCheck))
        gradient[gradient == i] = unset

        c = 0
        for index in toCheck.keys():
            value = gradient[index[0], index[1]]
            if value == unset:
                found = unfound

                # Find a surrounding value above 255
                for adj in adjacent:
                    if gradient[index[0] + adj[0], index[1] + adj[1]] >= maxMarker:
                        # This is a boundary point
                        if found != unfound:
                            # If this is a boundary between two different markers
                            if found != gradient[index[0] + adj[0], index[1] + adj[1]]:
                                gradient[index[0], index[1]] = boundary
                                break
                            else:
                                # if don't, dont change its value
                                continue


                        # If found change to that value
                        gradient[index[0], index[1]] = gradient[index[0] + adj[0], index[1] + adj[1]]
                        toCheck.remove((index[0], index[1]))

                        # Save which piece it borders on
                        found = gradient[index[0] + adj[0], index[1] + adj[1]]
                        c += 1

        print("Found:", c)

    # Prepare for return
    gradient[gradient == 0] = 9999
    gradient = gradient - 258
    gradient[gradient == (9999 - 258)] = 255

    return gradient.astype(np.uint8)