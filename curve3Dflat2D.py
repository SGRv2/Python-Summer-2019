import cv2
import numpy as np

class Line(object):
    def __init__(self, p1, p2):
        """
        For line y = m*x + c, calculate m and c parameters
        If the line is vertical, set "vertical" attr to True and save "x" position of the line
        """
        self.p1 = p1
        self.p2 = p2
        self.vertical = False
        self.fixed_x = None
        self.k = None
        self.b = None

        # cached angle props
        self.angle = None
        self.angle_cos = None
        self.angle_sin = None

        self.set_line_props(point1, point2)

    def is_vertical(self):
        return self.vertical

    def set_line_props(self, point1, point2):
        if point2[0] - point1[0]:
            self.k = float(point2[1] - point1[1]) / (point2[0] - point1[0])
            self.b = point2[1] - self.k * point2[0]

            k_normal = - 1 / self.k
        else:
            self.vertical = True
            self.fixed_x = point2[0]

            k_normal = 0

        self.angle = np.arctan(k_normal)
        self.angle_cos = np.cos(self.angle)
        self.angle_sin = np.sin(self.angle)

    def get_x(self, y):
        if self.is_vertical():
            return self.fixed_x
        else:
            return int(round(float(y - self.b) / self.k))

    def get_y(self, x):
        return self.k * x + self.b


class LabelUnwrapper(object):
    COL_COUNT = 30
    ROW_COUNT = 20
