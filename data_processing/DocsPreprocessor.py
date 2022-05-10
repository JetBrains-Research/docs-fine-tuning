import os
import re
from typing import List

import numpy as np
from bs4 import BeautifulSoup
from markdown import markdown
from tika import parser

from util import preprocess


class DocsPreprocessor:
    def __init__(self, files_path: str, extensions: List[str]):
        self.extensions = ["." + extension for extension in extensions]
        self.files = self.__collect_files(files_path)

    def preprocess_files(self):
        result = []
        for file_name in self.files:
            tokenized = self.read_and_preprocess(file_name)
            if isinstance(tokenized, float) == False and tokenized not in result:
                result.append(tokenized)
        return result

    def read_and_preprocess(self, file_name):
        text = DocsPreprocessor.__read_file(file_name)
        file_extension = os.path.splitext(file_name)[1][1:]
        tokenized = DocsPreprocessor.__preprocess(text, file_extension)
        return tokenized

    @staticmethod
    def __preprocess(text, format):
        if format == "md":
            is_html = bool(BeautifulSoup(text, "html.parser").find())
            text = re.sub(r"```[^\S\r\n]*[a-z]*\n.*?\n```", "", text, 0, re.DOTALL)
            text = DocsPreprocessor.__strip_html(text)
            if is_html == False:
                text = markdown(text)
        elif format == "html":
            text = DocsPreprocessor.__strip_html(text)

        return preprocess(text)

    @staticmethod
    def __read_file(file_name: str) -> str:
        if file_name.endswith(".pdf"):
            raw = parser.from_file(file_name)
            return raw["content"]

        with open(file_name, "r") as f:
            text = f.read()
        return text

    @staticmethod
    def __strip_html(text):
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

    def __should_preprocess(self, file_name):
        return np.any([file_name.endswith(extension) for extension in self.extensions])