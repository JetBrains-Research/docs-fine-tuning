import sys

from pathlib import Path
from tika import parser
from util import remove_noise
from util import tokenize_and_normalize


def get_text_from_pdf(file_name):
    raw = parser.from_file(file_name)
    return raw['content']


def main(pdf_docs):
    for i, doc in enumerate(pdf_docs):
        text = get_text_from_pdf(doc)
        text = remove_noise(text)
        text = tokenize_and_normalize(text)
        with Path(f'data/docs/doc_{i}.txt').open(mode='w') as f:
            f.write(str(text))


if __name__ == "__main__":
    main(sys.argv[1:])
