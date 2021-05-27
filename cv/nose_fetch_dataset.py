from face_fetch import fetch_and_save


source_dir = "../dataset/faces"
DEST_DIR = "../dataset/noses_out"


def main():
    fetch_and_save(source_dir, DEST_DIR, part="nose")


if __name__ == '__main__':
    main()
