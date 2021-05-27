from web_client import WebClient
from pathlib import Path
import asyncio
from bs4 import BeautifulSoup
import re


SAVE_DIR = "../dataset/cv/hd"

HD_IMG_NAME_PATTERN = "hd-{0}.[ext]"

INDEX_FROM = 2
START_FROM = 31

SOURCE_PATTERN = "https://papers.co/page/{0}/?cat=1&s=asian#038;s=asian"


def main():
    global SAVE_DIR, HD_IMG_NAME_PATTERN, INDEX_FROM

    print("Start fetching ...", end='')
    client = WebClient(init_async_session=False)
    i = 1
    count = 0
    while True:
        try:
            s = client.get_string(SOURCE_PATTERN.format(i))
        except:
            break

        soup = BeautifulSoup(s, 'html.parser')
        rec_list = soup.find(attrs={'class': 'postul'})
        list_items = rec_list.find_all(attrs={'class': 'postli'})

        for item in list_items:
            detail_url = item.find('a').attrs['href']
            print(f'\rAccessing {detail_url} ...', end="")

            if INDEX_FROM + count < START_FROM:
                print("Skipped")
                count += 1
                continue

            try:
                s2 = client.get_string(detail_url)
            except:
                print(f'\rFailed to access {detail_url}')
                continue

            soup2 = BeautifulSoup(s2, 'html.parser')
            a = soup2.find(attrs={'class': 'downloadbox'}).find_all('a')[-1]

            print(f"\rDownloading from {detail_url} ...", end="")
            client.save_content(a.attrs['href'], save_dir=SAVE_DIR,
                                file_name_pattern=HD_IMG_NAME_PATTERN.format(INDEX_FROM + count),
                                force_overwrite=True)
            count += 1
            print("done")

        i += 1

    client.close()
    print("done")


if __name__ == '__main__':
    # asyncio.run(main())
    main()
