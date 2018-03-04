import cv2
import os

LINKS_CSV = 'sheets.csv'

def main():
    with open(LINKS_CSV, 'r') as fp:
        i = 2000
        for i, line in enumerate(fp):
            line = line.strip()
            img = cv2.imread(line, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join('open-im-0', str(i + 2000), '.jpg'), img)

if __name__ == '__main__':
    main()
