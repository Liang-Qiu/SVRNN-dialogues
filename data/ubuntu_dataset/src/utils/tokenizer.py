import gzip
from nltk.tokenize import TreebankWordTokenizer

from .io_utils import read_ubuntu_threads


def tokenize_all(sentences):
    tokenizer = TreebankWordTokenizer()
    for sentence in sentences:
        for line in sentence:
            line[2] = tokenizer.tokenize(line[2])
    return sentences


def save(fn, data):
    with gzip.open(fn + '.gz', 'wb') as gf:
        for f in data:
            for line in f:
                text = '%s\t%s\t%s\n' % (line[0], line[1], ' '.join(line[2]))
                gf.writelines(text)
            gf.writelines('\n')


def main(argv):
    fn = argv.data.split('/')[-1]
    sentences = read_ubuntu_threads(argv.data)  # 1D: file, 2D: line
    save(fn, tokenize_all(sentences))
