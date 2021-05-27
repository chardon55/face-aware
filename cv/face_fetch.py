import face_detect
from pathlib import Path
import cv2 as cv
import numpy as np

source_dir = "../dataset/cv/hd"
DEST_DIR = "../dataset/cv_out/raw"


def fetch_and_save(source_dir, destination_dir, part="face"):
    print("Start fetching ...")
    source = Path(source_dir)
    dest = Path(destination_dir)
    if not dest.exists():
        dest.mkdir()

    i = 0

    for item in source.rglob("*"):
        if item.is_dir():
            continue

        img = cv.imread(str(item))

        print(f"\rDetecting {item}... ", end='')
        faces = face_detect.detect(img, detector=part)

        for face in faces:
            cropped = face_detect.crop_image(img, *tuple(face))

            print(f"\rSaving {item}...    \b\b\b", end="")
            cv.imwrite(str(dest / f"{i}.png"), cropped)
            i += 1

            print("done")


def main():
    fetch_and_save(source_dir, DEST_DIR)


if __name__ == '__main__':
    main()
