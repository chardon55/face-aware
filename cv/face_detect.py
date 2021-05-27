import numpy as np
import cv2 as cv
from pathlib import Path

from typing import List

HAARCASCADES = Path("./haarcascades")

detectors = None


def init_detectors():
    return {
        'face': cv.CascadeClassifier(str(HAARCASCADES / 'haarcascade_frontalface_default.xml')),
        'eye': cv.CascadeClassifier(str(HAARCASCADES / 'haarcascade_eye.xml')),
        'nose': cv.CascadeClassifier(str(HAARCASCADES / 'nose.xml')),
        'mouth': cv.CascadeClassifier(str(HAARCASCADES / 'mouth.xml'))
    }


def load_detector(detector: str):
    global detectors

    if detectors is None:
        detectors = init_detectors()

    return detectors[detector]


# def area_diff_percent(pos_vec1: tuple, pos_vec2: tuple):
#     m = np.array((pos_vec1, pos_vec2))
#     area_diff = 0
#
#     x_dist = np.min(np.array((m[0, 0] + m[0, 2], m[1, 0] + m[1, 2]))) - np.max(np.array((m[0, 0], m[1, 0])))
#     y_dist = np.min(np.array((m[0, 1] + m[0, 3], m[1, 1] + m[1, 3]))) - np.max(np.array((m[0, 1], m[1, 1])))
#     if x_dist > 0 and y_dist > 0:
#         area_diff = x_dist * y_dist
#
#     return np.max(np.array((area_diff / (m[0, 2] * m[0, 3]), area_diff / (m[1, 2] * m[1, 3]))))


def detect(image,
           detector="face",
           scale_factor=1.1, min_neighbors=5, min_size=(30, 30),
           unsharp_mask=True, edge_enhance=True):
    img = image.copy()
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    if unsharp_mask:
        gaussian_blurred = cv.GaussianBlur(gray, (3, 3), 5.0)
        gray = cv.addWeighted(gray, 2, gaussian_blurred, -1, 0)

    if edge_enhance:
        edge_img = cv.Canny(gray, 200, 350)
        gray = cv.addWeighted(gray, 1, edge_img, 0.07, 0)

    face_detector = load_detector(detector)
    rects = face_detector.detectMultiScale(gray,
                                           scaleFactor=scale_factor,
                                           minNeighbors=min_neighbors,
                                           minSize=min_size,
                                           flags=cv.CASCADE_SCALE_IMAGE)

    yield from [tuple(rect) for rect in rects]


def crop_image(image, start_x, start_y, length_x, length_y):
    return image[start_y:start_y+length_y, start_x:start_x+length_x]


class Rectangle(object):
    start_point: tuple
    end_point: tuple
    bgr_color: tuple
    thickness: int

    def __init__(self, start_point: tuple, length_point: tuple, rgb_color: tuple, thickness: int):
        self.start_point = start_point
        self.end_point = (start_point[0] + length_point[0], start_point[1] + length_point[1])
        self.bgr_color = rgb_color[::-1]
        self.thickness = thickness


def render(image, rectangles: List[Rectangle]):
    img = image.copy()

    for rect in rectangles:
        cv.rectangle(img,
                     rect.start_point, rect.end_point,
                     rect.bgr_color, rect.thickness)

    cv.imshow('window', img)
    cv.waitKey()


def main():
    image_path = "../dataset/cv/lena-0.png"

    img = cv.imread(image_path)

    size = img.shape

    max_size = (500, 500)
    print(size)

    if size[0] > max_size[0] or size[1] > max_size[1]:
        scale = min(max_size[0] / size[0], max_size[1] / size[1])
        img = cv.resize(img, (int(size[1] * scale), int(size[0] * scale)))

    # img1 = img.copy()
    #
    # font = cv.FONT_HERSHEY_COMPLEX
    #
    # cv.putText(img1, "Original", (50, 50), font, 2, (0, 0, 0))
    # cv.imshow('window', img1)
    # cv.waitKey()
    #
    # gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # cv.putText(gray, "Grayscale", (50, 50), font, 2, (255, 255, 255))
    # cv.imshow('window', gray)
    # cv.waitKey()

    rects = []

    faces = detect(img, detector="face")

    print("Faces:")
    for face in faces:
        print(face)
        rects.append(Rectangle(start_point=(face[0], face[1]),
                               length_point=(face[2], face[3]),
                               rgb_color=(0, 255, 0),
                               thickness=2))
        # cv.imshow('window', crop_image(img, *tuple(face)))
        # cv.waitKey()

    eyes = detect(img, detector="eye")
    print("Eyes:")
    for eye in eyes:
        print(eye)
        rects.append(Rectangle(start_point=(eye[0], eye[1]),
                               length_point=(eye[2], eye[3]),
                               rgb_color=(0, 0, 255),
                               thickness=2))

    # cv.putText(img, "Detection", (50, 50), font, 2, (0, 0, 0))
    render(img, rects)


if __name__ == '__main__':
    main()
