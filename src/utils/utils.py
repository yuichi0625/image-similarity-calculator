from typing import List, Union

import cv2
import numpy as np


def resize_if_exceeds(img: np.ndarray,
                      limit_length: int) -> np.ndarray:
    """縦横がlimit_lengthを超えた場合にリサイズする
    """
    ratio = limit_length / max(*img.shape[:2])
    if ratio < 1:
        img = cv2.resize(img, None, fx=ratio, fy=ratio)

    return img


def devide_into_groups(num_list: List[Union[int, float]],
                       diff: Union[int, float]) -> List[List[Union[int, float]]]:
    """
    例えば、
        num_list=[0, 0.3, 0.5, 0.8, 1, 3, 4, 5, 6, 9], diff=1
    の場合、以下を返す
        [[0, 0.3, 0.5, 0.8, 1], [3, 4, 5, 6], [9]]
    """
    num_list = sorted(num_list)

    groups = [[num_list[0]]]
    for num in num_list[1:]:
        group = groups[-1]
        if num - group[-1] <= diff:
            group.append(num)
        else:
            groups.append([num])

    return groups
