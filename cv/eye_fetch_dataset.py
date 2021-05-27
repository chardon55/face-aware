from face_fetch import fetch_and_save


source_dir = "../dataset/faces"
DEST_DIR = "../dataset/eyes_out"


def main():
    fetch_and_save(source_dir, DEST_DIR, part="eye")


if __name__ == '__main__':
    main()
