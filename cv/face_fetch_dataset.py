from face_fetch import fetch_and_save


source_dir = "../dataset/faces"
DEST_DIR = "../dataset/faces_out"


def main():
    fetch_and_save(source_dir, DEST_DIR)


if __name__ == '__main__':
    main()
