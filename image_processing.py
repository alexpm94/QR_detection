from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from settings import *


def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # resize the input image and preprocess it
    image = img_to_array(image)
    image = image.astype(IMAGE_DTYPE)
    image_org = image.copy()

    # return the processed image
    return image_org


def equalization_clahe(img_gray, rgb=False):
    """Equalization using CLAHE
    Args:
        img_gray (np.array): 2 dimensional array
        rgb (bool): Flag if RGB array

    Returns:
        image CLAHE applied

    """
    if rgb:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)

    img_gray = cv2.GaussianBlur(img_gray, (21, 21), 1)
    img_gray = np.uint8(img_gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(7, 7))
    return clahe.apply(img_gray)


def preprocess(image):
    """Make image binary
    Args:
        image (np.array): dimensional array in BGR

    Returns:
        image binary image
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_clahe = equalization_clahe(image_gray)
    binary = cv2.threshold(image_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return binary


def detect_edges(image):
    """Performs canny edge detector
    Args:
        image (np.array): 2 dimensional array
    Returns:
        edged (np.array): image containing edges
    """
    # In this case Hysteresis thresholds are ignored
    # beacuse a binary image is received
    edged = cv2.Canny(image, 0, 255)

    return edged


def detect_finder_patterns(edged):
    """Edged image
    Args:
        edged (np.array): 2 dimensional array

    Returns:
        list_patterns (np.array): Array of 3 elements containing the 3
                    finder patterns. None if less than 3 patterns found.
    """

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    list_patterns = []
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if 100 < area < 10000:
                list_patterns.append(approx)

    list_patterns = sorted(list_patterns, key=cv2.contourArea, reverse=True)[:3]

    if len(list_patterns) < 3:
        return None

    return list_patterns


def find_pattern_points(list_squares):
    """ Finds the center points of the given finder patterns

    Args:
        list_squares (list): List of finder patterns

    Returns:
        points (list): List of center points

    """
    # Find square center
    points = []
    for square in list_squares:
        xmin = min(square[:, :, 0])[0]
        xmax = max(square[:, :, 0])[0]

        ymin = min(square[:, :, 1])[0]
        ymax = max(square[:, :, 1])[0]

        point = np.array((xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2))
        points.append(point)

    return points


def find_nw_pattern(points):
    """ Finds the North West point, given a set of points

    Args:
        points (list): List of center points

    Returns:
        index_nw (int): Index of the nw found point

    """
    distances = [0, 0, 0]

    dist1 = np.linalg.norm(points[0] - points[1])
    dist2 = np.linalg.norm(points[0] - points[2])
    dist3 = np.linalg.norm(points[1] - points[2])

    distances[0] = dist1 + dist2
    distances[1] = dist1 + dist3
    distances[2] = dist2 + dist3

    index_nw = np.argmin(distances)

    return index_nw


def calculate_nw_rotation(nw_point, image_shape):
    """ Calculate rotation Matrix to bring nw point to the
        upper left corner.

    Args:
        nw_point (tuple): Point given (x,y) coordinates
        image_shape (tuple): Image shape (h,w)

    Returns:
        tuple: Rotation Matrix and correction angle

    """
    # Reference point
    ref_point = [0, 0]
    center_image = np.array(image_shape) // 2
    angle_ref = -np.arctan2(ref_point[0] - center_image[0], ref_point[1] - center_image[1]) * 180 / np.pi

    # Angle to nw point
    nw_angle = -np.arctan2(nw_point[0] - center_image[0], nw_point[1] - center_image[1]) * 180 / np.pi
    rotation_angle = nw_angle - angle_ref

    # Find Rotation Matrix
    (h, w) = image_shape
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)

    return M, rotation_angle


def calculate_correction_angle(points, image_shape):
    """Calculate the whole rotation given an image

    Args:
        points (list): Set of finder patterns center point
        image_shape (tuple): Image shape (h, w)

    Returns:
        correction angle (float): Angle to be rotated
    """
    index_nw = find_nw_pattern(points)
    M, rotation_angle = calculate_nw_rotation(points[index_nw], image_shape)

    # Transform center points
    transformed_points = []
    for point in points:
        np_point = np.array([point[0], point[1], 1])
        transformed_points.append(np.dot(M, np_point))

    # Calculate Correction Anlge
    # Get North West point
    nw_point = transformed_points.pop(index_nw)

    index_ne = 0
    if transformed_points[1][0] > transformed_points[0][0]:
        index_ne = 1

    # Get North East point
    ne_point = transformed_points.pop(index_ne)

    # Get Correction Angle
    correction_angle = np.arctan2(ne_point[1] - nw_point[1], ne_point[0] - nw_point[0]) * 180 / np.pi

    return rotation_angle + correction_angle


def rotate_image(image, angle):
    """Rotates image bya a given angle

    Args:
        image (np.array): Image array
        angle (float):  Rotation angle

    Returns:
        rotated_image (np.array)
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def correct_image(qr):
    """ Takes a QR image and fixes its orientation
    Args:
        qr (np.array): QR image, usually found by an AI detector

    Returns:
        oriented_qr (np.array): Rotated QR if valid constrains, otherwise
                                returns the original image
    """
    resized = cv2.resize(qr, (300, 300))
    binary = preprocess(resized)
    edged = detect_edges(binary)
    image_shape = edged.shape[:2]

    finder_patterns = detect_finder_patterns(edged)

    if not finder_patterns:
        return qr

    points = find_pattern_points(finder_patterns)
    angle = calculate_correction_angle(points, image_shape)

    return rotate_image(qr, angle)