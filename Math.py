import numpy as np



def line_gradient(line):
    x1, y1, x2, y2 = line

    # Check for a divide by 0 error
    if x2 - x1 == 0:
        return np.pi / 2

    return np.arctan((y2 - y1) / (x2 - x1))


def line_length(line):
    x1, y1, x2, y2 = line

    return np.linalg.norm(np.array([[x1, y1], [x2, y2]]))