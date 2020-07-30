import os
import re
from glob import glob
from itertools import combinations

import cv2
import numpy as np
from PIL import Image


def calc_similarities_combinatorially(img_dir, q, sim_scores, sim_src_paths, display_imgs, detect_min_length=500, display_width=300):
    """Calculate similarities and append results to each queue/list

    This is a function for threading in tkinter app.

    Args:
        img_dir (str): Directory path containing input images
        q (collections.deque): Deque containing progress percentages (0 - 100)
        sim_scores (List[float]): List containing similarity scores of every image pair
        sim_src_paths (List[List[str]]): List containing paths of every image pair matched to sim_scores
        display_imgs (List[PIL.Image]): List containing RGB display images
        detect_min_length (int, optional): Input images will be resized to this length if they are longer than it, defaults to 500
        display_width (int, optional): Result images will be resized to this width, defaults to 300.
    """
    # read all the images and resize them if it's too big
    img_paths = extract_image_paths(img_dir)
    imgs_dict = {img_path: resize_if_exceeds(cv2.imread(img_path), detect_min_length) for img_path in img_paths}

    # extract features from all the images
    detector = cv2.AKAZE_create()
    kp_des_dict = {img_path: detector.detectAndCompute(img, None) for img_path, img in imgs_dict.items()}

    # loop all the combinations
    comb_paths = combinations(img_paths, 2)
    num_combs = len(img_paths) * (len(img_paths) - 1) / 2
    bf = cv2.BFMatcher()
    num_done = 0
    for path1, path2 in comb_paths:
        img1 = imgs_dict[path1].copy()
        img2 = imgs_dict[path2].copy()
        kp1, des1 = kp_des_dict[path1]
        kp2, des2 = kp_des_dict[path2]

        good = match_features(img1, img2, des1, des2, bf)

        degree, scale = calc_degree_and_scale(good, kp1, kp2)

        rimg1, rimg2 = resize_by_scale(img1, img2, scale)
        rimg1, rimg2 = rotate_by_degree(rimg1, rimg2, degree)

        score, bbox, query_flag = match_template(rimg1, rimg2)

        # depict results and create an image for displaying
        if query_flag == 1:
            q_img, t_img = img1, img2
            q_path, t_path = path1, path2
        else:  # query_flat == 2
            q_img, t_img = img2, img1
            q_path, t_path = path2, path1
        q_img = draw_matched_bbox(q_img, degree, bbox)
        display_img = create_display_image(q_img, t_img, q_path, t_path, score, display_width)
        pil_display_img = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))

        # update queue and lists
        num_done += 1
        q.append(num_done / num_combs * 100)
        sim_scores.append(score)
        sim_src_paths.append([path1, path2])
        display_imgs.append(pil_display_img)


def extract_image_paths(img_dir):
    """Extract all the image paths in the given directory recursively

    Args:
        img_dir (str): Directory containing images

    Returns:
        List[str]: List containing image paths
    """
    regexp = re.compile('.+(jpg|jpeg|png)')
    img_paths = [path for path in glob(os.path.join(img_dir, '**'), recursive=True)
                 if regexp.search(path.lower()) and path.isascii()]
    return img_paths


def resize_if_exceeds(img, detect_min_length):
    """Resize input image if its width or height exceeds detect_min_length

    Args:
        img (np.ndarray): BGR image
        detect_min_length (int): Mininum acceptable length of the input image

    Returns:
        np.ndarray: BGR resized image
    """
    ratio = detect_min_length / max(*img.shape[:2])
    if ratio < 1:
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
    return img


def match_features(img1, img2, des1, des2, bf_matcher, ratio=0.8, num_matched=20):
    """Match features between two input images, and return most probable ones

    Args:
        img1 (np.ndarray): BGR input image 1
        img2 (np.ndarray): BGR input image 2
        des1 (np.ndarray): descriptors from img1
        des2 (np.ndarray): descriptors from img2
        bf_matcher (cv2.BFMatcher): Brute-force matcher object
        ratio (float, optional): Ratio used to extract good matched features, defaults to 0.8
        num_matched (int, optional): Number of matched features to return, defaults to 20

    Returns
        List[List[cv2.DMatch]]: Matched features between img1 and img2
    """
    # if no features were detected, detectAndCompute returns [] and None
    if des1 is None or des2 is None:
        return []

    matches = bf_matcher.knnMatch(des1, des2, k=2)
    # matches has to contain more than 1 element, which has shape=(2,)
    if not matches or not all([len(match) == 2 for match in matches]):
        return []

    good = [[m] for m, n in matches if m.distance < ratio * n.distance]
    good = sorted(good, key=lambda x: x[0].distance)[:num_matched]
    return good


def calc_degree_and_scale(matched, kp1, kp2, min_group_length=4):
    """Calculate degree and scale from matched features

    Args:
        matched (List[List[cv2.DMatch]]): Matched features between img1 and img2
        kp1 (List[cv2.KeyPoint]): Feature keypoints in img1
        kp2 (List[cv2.KeyPoint]): Feature keypoints in img2
        min_group_length (int, optional): Minimum devided group length, defaults to 4

    Returns:
        tuple: Tuple containing:
            rel_deg (float): Relative degree calculated by matched features
            dist_scale (float): Distance scale calculated by matched features
    """
    # matched has to contain more than 1 for calculating combinations
    if len(matched) < 2:
        return 0., 0.
    # shape=(num_matched, 2(q, t), 2(x, y))
    coords = [[kp1[match[0].queryIdx].pt, kp2[match[0].trainIdx].pt] for match in matched]
    # shape=(nC2, 2(pair), 2(q, t), 2(x, y))
    combs = np.array(list(combinations(coords, 2)))
    # shape=(nC2, 2(q, t), 2(x, y))
    diff_pairs = np.subtract(combs[:, 1], combs[:, 0])
    # shape=(nC2, 2(q, t))
    degs = np.arctan2(diff_pairs[..., 1], diff_pairs[..., 0]) * 180 / np.pi
    dists = np.sqrt(diff_pairs[..., 0] ** 2 + diff_pairs[..., 1] ** 2)
    # shape=(nC2,)
    rel_degs = np.subtract(degs[:, 0], degs[:, 1])
    with np.errstate(divide='ignore'):
        # handle warning and replace inf with 0
        dist_scales = dists[:, 0] / dists[:, 1]
        dist_scales[~ np.isfinite(dist_scales)] = 0
    # group relative degrees and extract the most longest one
    rel_deg_groups = devide_into_groups(sorted(rel_degs), 1)
    rel_deg_group = max(rel_deg_groups, key=len)
    # the length has to be more than min_group_length
    if len(rel_deg_group) < min_group_length:
        return 0., 0.
    # shape=(1,)
    else:
        rel_deg = calc_median(rel_deg_group)
        dist_scale = dist_scales[rel_degs == rel_deg][0]
        return rel_deg, dist_scale


def devide_into_groups(num_list, diff):
    """Devide list into groups whose adjacent differences are less than/equal to diff

    For example, suppose
        num_list=[0, 0.3, 0.5, 0.8, 1, 3, 4, 5, 6, 9], diff=1
    It returns
        [[0, 0.3, 0.5, 0.8, 1], [3, 4, 5, 6], [9]]

    Args:
        num_list (List[int/float]): List that will be devided
        diff (int/float): Acceptable difference in the same group

    Returns:
        List[List[int/float]]: Devided list
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


def calc_median(num_list):
    """Calculate median

    This returns an existing number even with a even number list.
    For example, suppose num_list=[1, 2, 3, 4]
        statistics.median(num_list) returns 2.5
        calc_median(num_list) returns 3

    Args:
        num_list (List[int/float]): List of numbers

    Returns:
        int/float: Median of the list
    """
    return num_list[len(num_list) // 2]


def resize_by_scale(img1, img2, scale):
    """Resize the bigger image for matching its scale with the smaller one

    Args:
        img1 (np.ndarray): BGR input image 1
        img2 (np.ndarray): BGR input image 2
        scale (float): Scale between input images (calculated by img1 / img2)

    Returns:
        tuple: Tuple containing:
            img1 (np.ndarray): img1, resized if scale > 1
            img2 (np.ndarray): img2, resized if scale < 1
    """
    if scale == 0:
        return img1, img2
    elif scale > 1:
        scale = 1 / scale
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)
    else:
        img2 = cv2.resize(img2, None, fx=scale, fy=scale)
    return img1, img2


def rotate_by_degree(img1, img2, degree):
    """Rotate the bigger image

    Because the smaller image will be used as a template in template-matching,
    it won't be able to include black padding areas.

    Args:
        img1 (np.ndarray): BGR input image 1
        img2 (np.ndarray): BGR input image 2
        degree (float): Relative degree between two images

    Returns:
        tuple: Tuple containing:
            img1 (np.ndarray): img1, rotated if img1 is bigger than img2
            img2 (np.ndarray): img2, rotated if img2 is bigger than img1
    """
    if degree == 0:
        return img1, img2
    elif img1.size >= img2.size:
        h, w = img1.shape[:2]
        trans = cv2.getRotationMatrix2D((w//2, h//2), degree, 1.0)
        img1 = cv2.warpAffine(img1.copy(), trans, (w, h))
    else:
        h, w = img2.shape[:2]
        trans = cv2.getRotationMatrix2D((w//2, h//2), degree, 1.0)
        img2 = cv2.warpAffine(img2.copy(), trans, (w, h))
    return img1, img2


def match_template(img1, img2):
    """Match template using cv2.matchTemplate

    The bigger image will be the query and the smaller one is the template.

    Args:
        img1 (np.ndarray): BGR input image 1
        img2 (np.ndarray): BGR input image 2

    Returns:
        tuple: Tuple containing:
            clipped max_val (float): Similarity score between two input images
                                     This is clipped because cv2.matchTemplate actually computes correlation coefficient (-1 to 1)
            bbox (Tuple[int]): Bounding box (xmin, ymin, xmax, ymax)
            flag (int): Flat representing which image is the query one
                        {1: img1, 2: img2} is the query image
    """
    # bigger image will be the query one
    if img1.size >= img2.size:
        q_img = img1.copy()
        t_img = img2.copy()
        flag = 1
    else:
        q_img = img2.copy()
        t_img = img1.copy()
        flag = 2

    # template image has to be smaller than query one
    h_ratio = q_img.shape[0] / t_img.shape[0]
    w_ratio = q_img.shape[1] / t_img.shape[1]
    ratio = min(h_ratio, w_ratio)
    if ratio < 1:
        t_img = cv2.resize(t_img, None, fx=ratio, fy=ratio)

    res = cv2.matchTemplate(q_img, t_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    h, w = t_img.shape[:2]
    bbox = (*max_loc, max_loc[0] + w, max_loc[1] + h)
    return np.clip(max_val, 0., 1.), bbox, flag


def draw_matched_bbox(q_img, degree, bbox):
    """Draw bounding box on the query image

    Bounding box could be rotated, so rerotate it when depicting.

    Args:
        q_img (np.ndarray): BGR query image
        degree (float): Relative degree between two images
        bbox (Tuple(int)): Bounding box (xmin, ymin, xmax, ymax)

    Returns:
        np.ndarray: BGR query image with bbox depicted
    """
    xmin, ymin, xmax, ymax = bbox

    h, w = q_img.shape[:2]
    rh = h // 2
    rw = w // 2
    coords = np.array([[xmin - rw, ymin - rh, 0], [xmax - rw, ymin - rh, 0],
                       [xmax - rw, ymax - rh, 0], [xmin - rw, ymax - rh, 0]])
    trans = cv2.getRotationMatrix2D((rw, rh), -degree, 1.0)
    trans_coords = np.dot(trans, coords.T).T.astype(np.int32) + np.array([rw, rh])

    for i, j in [[0, 1], [1, 2], [2, 3], [3, 0]]:
        cv2.line(q_img, tuple(trans_coords[i]), tuple(trans_coords[j]), (0, 0, 255), 2)
    return q_img


def create_display_image(q_img, t_img, q_path, t_path, score, display_width=300):
    """Create image for displaying

    Args:
        q_img (np.ndarray): BGR query image with bbox depicted
        t_img (np.ndarray): BGR template image
        q_path (str): Path of q_img
        t_path (str): Path of t_img
        score (float): Similarity score between q_img and t_img
        display_width (int, optional): Display image will be resized to this width, defaults to 300

    Returns:
        np.ndarray: BGR image with bbox, score and paths depicted
    """
    max_height = max(q_img.shape[0], t_img.shape[0])
    if q_img.shape[0] > t_img.shape[0]:
        t_img = np.vstack((t_img, np.zeros((max_height - t_img.shape[0], *t_img.shape[1:]), dtype=np.uint8)))
    else:
        q_img = np.vstack((q_img, np.zeros((max_height - q_img.shape[0], *q_img.shape[1:]), dtype=np.uint8)))
    img = np.hstack((q_img, t_img))
    img = cv2.resize(img, (display_width, int(img.shape[0] * display_width / img.shape[1])))
    img = np.vstack((np.full((78, *img.shape[1:]), (255, 255, 255), dtype=np.uint8), img))

    color = (0, 0, 0)
    line_type = cv2.LINE_AA
    duplex_font = cv2.FONT_HERSHEY_DUPLEX
    simplex_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'Score: {score:.2f}', (0, 21), duplex_font, 0.8, color, 1, line_type)
    cv2.putText(img, f'  left : {os.path.basename(q_path)}', (0, 44), simplex_font, 0.65, color, 1, line_type)
    cv2.putText(img, f'  right: {os.path.basename(t_path)}', (0, 69), simplex_font, 0.65, color, 1, line_type)
    return img
