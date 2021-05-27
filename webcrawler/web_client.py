import requests
import re
from pathlib import Path
import aiohttp
import aiofiles


def resolve_file_path(file_name_pattern, force_overwrite, save_dir, url, url_contains_extension):
    ext = re.split(r"\.", url)[-1]
    if url_contains_extension:
        file_name = re.sub(r"\[ext\]", ext, file_name_pattern)
    else:
        file_name = file_name_pattern
    file = (Path(save_dir) / file_name)
    if file.exists() and not force_overwrite:
        raise FileExistsError()
    return file


class WebClient:
    client_session = None

    def __init__(self, headers=None, init_async_session=True):
        if headers is None:
            headers = {}

        self.headers = headers
        if init_async_session:
            self.__init_client_session()

    def close(self):
        self.client_session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client_session.close()

    def requests_access(self, url: str, params=None, method='get', override_encoding=True):
        response = requests.request(method, url, params=params, headers=self.headers)
        response.raise_for_status()
        if override_encoding:
            response.encoding = response.apparent_encoding

        return response

    def __init_client_session(self):
        if self.client_session is None:
            self.client_session = aiohttp.ClientSession(headers=self.headers)

    async def request_access_async(self, url: str, params=None, method='get'):
        # self.__init_client_session()
        response = await self.client_session.request(method, url, params=params)
        response.raise_for_status()

        return response

    def get_string(self, url: str, params=None) -> str:
        return self.requests_access(url, method='get', params=params).text

    async def get_string_async(self, url: str, params=None, encoding=None) -> str:
        response = await self.request_access_async(url, method='get', params=params)
        return await response.text(encoding)

    def get_content(self, url: str):
        return self.requests_access(url, method='get', override_encoding=False).content

    async def get_content_async(self, url: str):
        response = await self.request_access_async(url, method='get')
        return await response.read()

    def save_content(self, url: str, save_dir: str, file_name_pattern: str,
                     url_contains_extension=True,
                     force_overwrite=False) -> None:
        b = self.get_content(url)

        file = resolve_file_path(file_name_pattern, force_overwrite, save_dir, url, url_contains_extension)

        with open(str(file), 'wb') as f:
            f.write(b)

    async def save_content_async(self, url: str, save_dir: str, file_name_pattern: str,
                                 url_contains_extension=True,
                                 force_overwrite=False):
        content = await self.get_content_async(url)

        file = resolve_file_path(file_name_pattern, force_overwrite, save_dir, url, url_contains_extension)

        async with aiofiles.open(str(file), 'wb') as f:
            await f.write(content)
