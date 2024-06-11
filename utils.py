import os
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
import cv2
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


@dataclass
class Range:
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class ColorRange:
    # 色の範囲を定義
    blue: Range
    yellow_strictly: Range
    yellow_loosely: Range

COLOR_RANGE = ColorRange(
    blue=Range(np.array([90, 50, 50]), np.array([130, 255, 255])),
    yellow_strictly=Range(np.array([20, 100, 100]), np.array([30, 255, 255])),
    yellow_loosely=Range(np.array([0, 100, 100]), np.array([100, 255, 255])),
)


def filter_blue(hsv: np.ndarray):
    # 青色のマスクを作成する
    blue_mask = cv2.inRange(hsv, COLOR_RANGE.blue.lower, COLOR_RANGE.blue.upper)
    # 青色の部分を除外するために、青色のマスクを反転する
    blue_mask_inv = cv2.bitwise_not(blue_mask)
    hsv = cv2.bitwise_and(hsv, hsv, mask=blue_mask_inv)
    return hsv


def focus_yellow(image: np.ndarray, hsv: np.ndarray, lower_yellow: np.ndarray, upper_yellow: np.ndarray):
    # マスクを作成する
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # マスクを適用して黄色の部分を抽出する
    yellow_only = cv2.bitwise_and(image, image, mask=yellow_mask)
    return yellow_only


def to_pil(bgr_image: np.ndarray, div: float = 8) -> Image.Image:
    shape = bgr_image.shape
    shape = (shape[1] // div, shape[0] // div)
    return Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)).convert("RGB").resize(shape)



def run_focusing_yellow(image_origin: np.ndarray, hsv_origin: np.ndarray, color_range: ColorRange):
    image = image_origin.copy()
    hsv = hsv_origin.copy()
    hsv = filter_blue(hsv)
    yellow_only_strict = focus_yellow(image, hsv, color_range.yellow_strictly.lower, color_range.yellow_strictly.upper)

    image = image_origin.copy()
    hsv = hsv_origin.copy()
    hsv = filter_blue(hsv)
    yellow_only_loose = focus_yellow(image, hsv, color_range.yellow_loosely.lower, color_range.yellow_loosely.upper)

    diff = yellow_only_loose - yellow_only_strict
    return yellow_only_strict, yellow_only_loose, diff


#-------------------------------- 

def draw_from_binary_mask(img: np.ndarray, mask: np.ndarray, color=np.array([255, 0, 0]), alpha=0.3) -> np.ndarray:
    # マスク領域に色を適用
    assert img.shape[:2] == mask.shape[:2]

    # マスクがTrueの部分だけを取り出す
    indices = np.where(mask)

    # 対応する画像の部分に色を適用
    img[indices[0], indices[1], :] = (img[indices[0], indices[1], :] * (1 - alpha) + color * alpha).astype(np.uint8)

    return img


def draw_bbox(image: np.ndarray, x, y, w, h, color=(0, 0, 255), thickness=2):
    """
    画像にbboxを描画する関数

    Parameters:
    - image: 画像 (numpy.ndarray)
    - x: bboxの左上のx座標
    - y: bboxの左上のy座標
    - w: bboxの幅
    - h: bboxの高さ
    - color: bboxの色 (BGR形式)
    - thickness: bboxの線の太さ

    Returns:
    - 画像 (numpy.ndarray) にbboxが描画されたもの
    """
    # 左上の座標
    top_left = (int(x), int(y))
    # 右下の座標
    bottom_right = (int(x + w), int(y + h))
    # bboxを描画
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image

#-------------------------------- 

def get_weak_reflection_parts(origin_image_p, dsrnet_s_test_l_p, dsrnet_s_test_r_p, dsrnet_s_test_rr_p, obj_mask=None, does_draw_bbox=False):
    origin_image = cv2.imread(origin_image_p)
    dsrnet_s_test_l = cv2.imread(dsrnet_s_test_l_p)
    dsrnet_s_test_r = cv2.imread(dsrnet_s_test_r_p)
    dsrnet_s_test_rr = cv2.imread(dsrnet_s_test_rr_p)

    origin_image_hsv = cv2.cvtColor(origin_image, cv2.COLOR_BGR2HSV)

    # 黄色部分に絞る (optional) (絞りすぎると、上納膜のような白色透明の部分が残らなくなるので注意)
    yellow_loosely_lower = np.array([0, 50, 160])
    yellow_loosely_upper = np.array([100, 255, 255])
    origin_image = focus_yellow(origin_image, origin_image_hsv, yellow_loosely_lower, yellow_loosely_upper)

    # diff = origin_image - dsrnet_s_test_l
    diff = dsrnet_s_test_l

    # 対象画像をバイナリ化し、マスク領域を抽出
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    Image.fromarray(mask)

    masked = cv2.bitwise_and(origin_image, origin_image, mask=mask)

    # 青色の部分を除外するために、青色のマスクを反転する
    blue_mask = cv2.inRange(origin_image_hsv, COLOR_RANGE.blue.lower, COLOR_RANGE.blue.upper)
    blue_mask_inv = cv2.bitwise_not(blue_mask)
    masked = cv2.bitwise_and(masked, masked, mask=blue_mask_inv)

    # masked = np.where(masked == 255, 0, masked)
    masked = np.where(masked == 0, 255, masked)

    # obj binary mask to bbox; draw ground truth bbox
    if obj_mask is not None:
        obj_mask = cv2.resize(obj_mask, (origin_image.shape[1], origin_image.shape[0]))
        obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(obj_mask)
        M = 10
        x, y, w, h = x - M, y - M, w + 2 * M, h + 2 * M

        if does_draw_bbox:
            masked = draw_bbox(masked, x, y, w, h, color=(0, 0, 255), thickness=2)

    return origin_image, dsrnet_s_test_l, mask, masked