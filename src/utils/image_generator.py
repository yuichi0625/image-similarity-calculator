import os
from typing import List

import cv2
import numpy as np
from PIL import Image

import config as c
from .utils import resize_if_exceeds


def generate_display_images(sim_img_paths: List[List[str]],
                            sim_scores: List[float],
                            ov_coords: List[List[List[int]]]):
    """GUIで描画する画像を作成する
    """
    # sim_scoresに合わせてソート
    sorted_items = sorted(
        zip(sim_img_paths, sim_scores, ov_coords), key=lambda x: x[1], reverse=True)
    sorted_items = sorted_items[:c.NUM_DISPLAY]

    for (query_path, templt_path), score, coords in sorted_items:
        query = resize_if_exceeds(cv2.imread(query_path), c.LIMIT_LENGTH)
        templt = resize_if_exceeds(cv2.imread(templt_path), c.LIMIT_LENGTH)

        # queryに座標を描画する
        for i, j in [[0, 1], [1, 2], [2, 3], [3, 0]]:
            cv2.line(query, tuple(coords[i]), tuple(coords[j]), (0, 0, 255), 2)

        # queryとtempltの高さを合わせる
        height = max(query.shape[0], templt.shape[0])
        if query.shape[0] > templt.shape[0]:
            templt = np.vstack((templt, np.zeros((height - templt.shape[0], *templt.shape[1:]), dtype=np.uint8)))
        else:
            query = np.vstack((query, np.zeros((height - query.shape[0], *query.shape[1:]), dtype=np.uint8)))

        # queryとtempltを横に繋げる
        img = np.hstack((query, templt))
        img = cv2.resize(img, (c.DISPLAY_IMAGE_WIDTH, int(img.shape[0] * c.DISPLAY_IMAGE_WIDTH / img.shape[1])))

        # 画像の上部に空白を加え、そこに類似度とパスを記載する
        color = (0, 0, 0)
        line_type = cv2.LINE_AA
        duplex_font = cv2.FONT_HERSHEY_DUPLEX
        simplex_font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.vstack((np.full((78, *img.shape[1:]), (255, 255, 255), dtype=np.uint8), img))
        cv2.putText(img, f'Score: {score:.2f}', (0, 21), duplex_font, 0.8, color, 1, line_type)
        cv2.putText(img, f'  left : {os.path.basename(query_path)}', (0, 44), simplex_font, 0.65, color, 1, line_type)
        cv2.putText(img, f'  right: {os.path.basename(templt_path)}', (0, 69), simplex_font, 0.65, color, 1, line_type)

        # tkinterで描画するためにPIL.Imageに変換する
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        yield pil_img
