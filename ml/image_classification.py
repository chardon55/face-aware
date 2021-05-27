from pathlib import Path
import cv2 as cv


source_dir = Path("../dataset/faces_out/")
destination_dir = Path("../dataset/ml/faces_class/")

classifications = {
    "100x100": (100, 100),
    "200x200": (200, 200),
    "300x300": (300, 300),
    "500x500": (500, 500),
    "700x700": (700, 700),
    "800x800": (800, 800),
    "1000x1000": (1000, 1000),
    "1500x1500": (1500, 1500),
    "2000x2000": (2000, 2000),
    "3000x3000": (3000, 3000),
    "4000x4000": (4000, 4000),
    "5000x5000": (5000, 5000),
}


def classify():
    global source_dir, destination_dir, classifications

    if not source_dir.is_dir():
        return

    counts = {}

    print("Start classifying... ", end='')

    for key in classifications.keys():
        target_dir = (destination_dir / key)

        if not target_dir.exists() or not target_dir.is_dir():
            target_dir.mkdir()

        counts[key] = 0

    for file in source_dir.rglob('*'):
        img = cv.imread(str(file))

        print(f"\rClassifying {str(file)}... ", end='')

        for key, val in classifications.items():
            if img.shape[0] >= val[0] and img.shape[1] >= val[1]:
                cv.imwrite(str(destination_dir / f"{key}/{counts[key]}.png"), cv.resize(img, dsize=val))
                counts[key] += 1
                print(f"{key} ", end='')

        print("done")


def main():
    classify()


if __name__ == '__main__':
    main()
