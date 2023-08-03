import math
import numpy as np

from utils import normalize

# this code is from https://gist.github.com/awesomebytes/7ccbd396511db71d0a51341569fa95fa


"""
Stuff to compute an orientation quaternion
from three points in the space (that represent a plane,
the quaternion represents the orientation of the plane).
Author: Sammy Pfeiffer <Sammy.Pfeiffer@student.uts.edu.au>
"""


def quaternion_from_forward_and_up_vectors(forward, up):
    """
    Inspired in the Unity LookRotation Quaternion function,
    https://answers.unity.com/questions/467614/what-is-the-source-code-of-quaternionlookrotation.html
    this returns a quaternion from two orthogonal vectors (x, y, z)
    representing forward and up.
    """
    v0 = normalize(forward)
    v1 = normalize(np.cross(normalize(up), v0))
    v2 = np.cross(v0, v1)
    m00, m01, m02 = v1
    m10, m11, m12 = v2
    m20, m21, m22 = v0

    num8 = (m00 + m11) + m22

    if num8 > 0.0:
        num = math.sqrt(num8 + 1.0)
        w = num * 0.5
        num = 0.5 / num
        x = (m12 - m21) * num
        y = (m20 - m02) * num
        z = (m01 - m10) * num
        return x, y, z, w

    if (m00 >= m11) and (m00 >= m22):
        num7 = math.sqrt(((1.0 + m00) - m11) - m22)
        num4 = 0.5 / num7
        x = 0.5 * num7
        y = (m01 + m10) * num4
        z = (m02 + m20) * num4
        w = (m12 - m21) * num4
        return x, y, z, w

    if m11 > m22:
        num6 = math.sqrt(((1.0 + m11) - m00) - m22)
        num3 = 0.5 / num6
        x = (m10 + m01) * num3
        y = 0.5 * num6
        z = (m21 + m12) * num3
        w = (m20 - m02) * num3
        return x, y, z, w

    num5 = math.sqrt(((1.0 + m22) - m00) - m11)
    num2 = 0.5 / num5
    x = (m20 + m02) * num2
    y = (m21 + m12) * num2
    z = 0.5 * num5
    w = (m01 - m10) * num2
    return x, y, z, w
