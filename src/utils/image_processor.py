from itertools import combinations
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import config as c
from .utils import devide_into_groups, resize_if_exceeds


def calc_sim(img_paths: List[str],
             sim_img_paths: List[List[str]],
             sim_scores: List[float],
             ov_coords: List[List[List[int]]]):
    # 画像をすべて読み込む
    # 特徴量抽出の計算量を減らすため、大きすぎる画像はリサイズしておく
    path2img = {img_path: resize_if_exceeds(cv2.imread(img_path), c.LIMIT_LENGTH)
                for img_path in tqdm(img_paths)}

    # 事前に画像の特徴量を取得する
    detector = cv2.AKAZE_create()
    path2kp_des = {img_path: detector.detectAndCompute(img, None)
                   for img_path, img in tqdm(path2img.items())}

    bf = cv2.BFMatcher()
    for path1, path2 in combinations(img_paths, r=2):
        img1 = path2img[path1].copy()
        img2 = path2img[path2].copy()
        kp1, des1 = path2kp_des[path1]
        kp2, des2 = path2kp_des[path2]

        # img1とimg2の特徴量のペアを取得する
        matched: List[List[cv2.DMatch]] = match_features(img1, img2, des1, des2, bf)

        # 特徴量のペアをもとに、2画像の相対的な角度と縮尺を計算する
        degree, scale = calc_degree_and_scale(matched, kp1, kp2)

        # 相対的な角度と縮尺をもとに、画像を補正する
        rimg1, rimg2 = resize_by_scale(img1, img2, scale)
        rimg1, rimg2 = rotate_by_degree(rimg1, rimg2, degree)

        # query: 大きい画像, templt: 小さい画像
        if img1.size >= img2.size:
            query, templt = rimg1, rimg2
            query_path, templt_path = path1, path2
        else:
            query, templt = rimg2, rimg1
            query_path, templt_path = path2, path1

        # 類似度を計算する
        # coordsはtempltがqueryに重なった部分の座標
        score, coords = match_template(query, templt)

        # coordsを回転前の座標に戻す
        coords = rerotate_coords(coords, query, degree)

        # 入力引数を更新する
        sim_img_paths.append([query_path, templt_path])
        sim_scores.append(score)
        ov_coords.append(coords)


def match_features(img1: np.ndarray,
                   img2: np.ndarray,
                   des1: np.ndarray,
                   des2: np.ndarray,
                   bf_matcher: cv2.BFMatcher,
                   ratio: float = 0.8,
                   num_return: int = 20) -> List[List[cv2.DMatch]]:
    """特徴量マッチングを行い、確率の高いものを返す
    """
    # 特徴量が検出されていない場合、desはNone
    if des1 is None or des2 is None:
        return []

    matches: List[List[cv2.DMatch]] = bf_matcher.knnMatch(des1, des2, k=2)
    # matchesの各リストが要素を2つ持っている必要あり
    if not matches or not all(len(match) == 2 for match in matches):
        return []

    good: List[List[cv2.DMatch]] = [[m] for m, n in matches if m.distance < ratio * n.distance]
    good = sorted(good, key=lambda x: x[0].distance)
    good = good[:num_return]
    return good


def calc_degree_and_scale(matched: List[List[cv2.DMatch]],
                          kp1: List[cv2.KeyPoint],
                          kp2: List[cv2.KeyPoint],
                          min_group_length: int = 4) -> Tuple[float, float]:
    """特徴量のペアから相対的な角度と縮尺を計算する
    """
    # 特徴量のペアが2個以上ないと、以降の計算ができないため例外とする
    if len(matched) < 2:
        return 0., 0.

    # https://qiita.com/grouse324/items/74988134a9073568b32d
    # shape=(num_matched, 2(query, train), 2(x座標, y座標))
    coords = [[kp1[match[0].queryIdx].pt, kp2[match[0].trainIdx].pt] for match in matched]
    # shape=(nC2, 2(ペア), 2(query, train), 2(x座標, y座標))
    combs = np.array(list(combinations(coords, 2)))
    # shape=(nC2, 2(query, train), 2(x座標, y座標))
    diff_pairs = np.subtract(combs[:, 1], combs[:, 0])
    # shape=(nC2, 2(query, train))
    degs: np.ndarray = np.arctan2(diff_pairs[..., 1], diff_pairs[..., 0]) * 180 / np.pi
    dists = np.sqrt(diff_pairs[..., 0] ** 2 + diff_pairs[..., 1] ** 2)
    # shape=(nC2,)
    rel_degs = np.subtract(degs[:, 0], degs[:, 1])
    with np.errstate(divide='ignore'):
        # warning対策とinfを0に置き換える
        scales = dists[:, 0] / dists[:, 1]
        scales[~ np.isfinite(scales)] = 0

    # 相対的な角度にはおかしな値も混ざっていることが多い
    # そのため、外れ値を除外する
    rel_deg_groups = devide_into_groups(rel_degs.tolist(), diff=1)
    rel_deg_group = max(rel_deg_groups, key=len)
    # 取得できた相対的な角度が少なすぎる場合は例外とする
    if len(rel_deg_group) < min_group_length:
        return 0., 0.
    else:
        rel_deg = rel_deg_group[len(rel_deg_group) // 2]
        scale = scales[rel_degs == rel_deg][0]
        return rel_deg, scale


def resize_by_scale(img1: np.ndarray,
                    img2: np.ndarray,
                    scale: float,
                    limit: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """scaleに合わせて大きい方の画像をリサイズする
    """
    # 特徴量によってscaleがおかしくなることがある
    # scaleが0, 1, >10, <1/10の場合は無視する
    if scale == 0 or scale == 1 or scale > limit or scale < 1 / limit:
        return img1, img2

    if scale > 1:
        scale = 1 / scale
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)
    else:
        img2 = cv2.resize(img2, None, fx=scale, fy=scale)

    return img1, img2


def rotate_by_degree(img1: np.ndarray,
                     img2: np.ndarray,
                     degree: float) -> Tuple[np.ndarray, np.ndarray]:
    """degreeに合わせて大きい方の画像を回転する
    """
    if degree == 0:
        return img1, img2

    if img1.size >= img2.size:
        h, w = img1.shape[:2]
        trans = cv2.getRotationMatrix2D((w//2, h//2), degree, 1.0)
        img1 = cv2.warpAffine(img1.copy(), trans, (w, h))
    else:
        h, w = img2.shape[:2]
        trans = cv2.getRotationMatrix2D((w//2, h//2), degree, 1.0)
        img2 = cv2.warpAffine(img2.copy(), trans, (w, h))

    return img1, img2


def match_template(query: np.ndarray,
                   templt: np.ndarray) -> Tuple[float, List[int]]:
    """テンプレートマッチングを行う
    """
    # 縦横ともに長さがtemplt < queryになるようにする
    ratio = min(
        query.shape[0] / templt.shape[0],
        query.shape[1] / templt.shape[1])
    if ratio < 1:
        templt = cv2.resize(templt, None, fx=ratio, fy=ratio)

    # テンプレートマッチングを行う
    res = cv2.matchTemplate(query, templt, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    # templtがqueryに重なる部分の座標を取得する
    h, w = templt.shape[:2]
    coords: List[int] = (*max_loc, max_loc[0] + w, max_loc[1] + h)  # (xmin, ymin, xmax, ymax)

    return float(np.clip(max_val, 0., 1.)), coords


def rerotate_coords(coords: List[int],
                    img: np.ndarray,
                    degree: float) -> List[List[int]]:
    """coordsをdegreeだけ逆回転する
    """
    h, w = img.shape[:2]
    rh, rw = h // 2, w // 2
    xmin, ymin, xmax, ymax = coords

    coords_ = np.array([
        [xmin - rw, ymin - rh, 0], [xmax - rw, ymin - rh, 0],
        [xmax - rw, ymax - rh, 0], [xmin - rw, ymax - rh, 0]])
    matrix = cv2.getRotationMatrix2D((rw, rh), -degree, 1.0)
    rot_coords = np.dot(matrix, coords_.T).T.astype(np.int32) + np.array([rw, rh])

    return rot_coords.tolist()
