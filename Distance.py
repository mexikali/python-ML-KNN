import numpy as np
import math


class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):

        if x is None or y is None:
            return float('inf')

        ab = 0
        a2 = 0
        b2 = 0

        for i in range(len(x)):
            ab += (x[i] * y[i])
            a2 += (x[i] ** 2)
            b2 += (y[i] ** 2)

        if a2 == 0 or b2 == 0:
            return float('inf')
        else:
            return 1 - (ab / (math.sqrt(a2) * math.sqrt(b2)))

    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        c = 0
        for i in range(len(x)):
            c += (x[i]-y[i])**p
        return c**(1/p)

    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1):
        diff = np.array(x) - np.array(y)
        return np.sqrt(np.dot(np.dot(diff.T, S_minus_1), diff))
