import os
import re
from typing import List, Union

import numpy as np
import rst2txt
from bs4 import BeautifulSoup
from docutils.core import publish_string
from markdown import markdown
from tika import parser

from util import preprocess, Section, Corpus, NumpyNaN


class DocsPreprocessor:
    def __init__(self, files_path: str, extensions: List[str]):
        self.extensions = [extension if extension.startswith(".") else "." + extension for extension in extensions]
        self.files = self.__collect_files(files_path)

    def preprocess_files(self) -> List[str]:
        result = []
        for file_name in self.files:
            if file_name.endswith("index.rst"):
                continue
            tokenized = DocsPreprocessor.read_and_preprocess(file_name)
            if not isinstance(tokenized, float) and tokenized not in result:
                result += tokenized
        return result

    @staticmethod
    def read_and_preprocess(file_name: str) -> List[str]:
        text = DocsPreprocessor.__read_file(file_name)
        file_extension = os.path.splitext(file_name)[1][1:]
        tokenized = DocsPreprocessor.__preprocess(text, file_extension)
        return tokenized

    @staticmethod
    def __preprocess(text, file_format: str) -> Union[List[str], NumpyNaN]:
        if file_format == "md":
            is_html = bool(BeautifulSoup(text, "html.parser").find())
            text = re.sub(r"```[^\S\r\n]*[a-z]*\n.*?\n```", "", text, 0, re.DOTALL)
            text = DocsPreprocessor.__strip_html(text)
            if is_html is False:
                text = markdown(text)
        elif file_format == "html":
            text = DocsPreprocessor.__strip_html(text)
        elif file_format == "rst":
            try:
                text = str(publish_string(source=text, writer=rst2txt.Writer()))
            except (RuntimeError, Exception):
                return np.nan

        return preprocess(text)

    @staticmethod
    def __read_file(file_name: str) -> str:
        if file_name.endswith(".pdf"):
            raw = parser.from_file(file_name)
            return raw["content"]

        with open(file_name, "r") as in_f:
            text = in_f.read()
        return text

    @staticmethod
    def __strip_html(text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def __collect_files(self, cur_dir: str) -> List[str]:
        if os.path.isfile(cur_dir) and self.__should_preprocess(cur_dir):
            return [cur_dir]

        result = []
        for file in os.listdir(cur_dir):
            path = os.path.join(cur_dir, file)
            if os.path.isdir(path):
                result += self.__collect_files(path)
            elif self.__should_preprocess(file):
                result.append(path)
        return result

    def __should_preprocess(self, file_name: str) -> bool:
        return any([file_name.endswith(extension) for extension in self.extensions])
