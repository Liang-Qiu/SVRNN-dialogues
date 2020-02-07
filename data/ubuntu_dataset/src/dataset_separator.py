import gzip
import argparse
import io
import os
from pathlib import Path
import numpy as np


def save(fn, data):
    with gzip.open(fn + '.gz', 'wb') as gf:
        with io.TextIOWrapper(gf, encoding='utf-8') as enc:
            for sents in data:
                for sent in sents:
                    enc.writelines("\t".join(sent) + '\n')
                enc.writelines('\n')


def main(argv):
    print('\nDATASET SEPARATION START')

    dataset = []
    thread = []

    path = Path(argv.data)
    os.chdir(path.parent)

    with gzip.open(argv.data, 'rb') as gf:
        with io.TextIOWrapper(gf, encoding='utf-8') as dec:
            for line in dec:
                line = line.rstrip().split("\t")
                if len(line) < 6:
                    if thread:
                        dataset.append(thread)
                        thread = []
                else:
                    thread.append(line)

    print('Threads: %d' % len(dataset))

    cand_indices = list(range(len(dataset)))
    np.random.shuffle(cand_indices)
    dataset = [dataset[i] for i in cand_indices]

    folded = len(dataset) / 20
    train_data = dataset[:int(folded * 18)]
    dev_data = dataset[int(folded * 18):int(folded * 19)]
    test_data = dataset[int(folded * 19):]

    print('Train: %d\tDev: %d\tTest: %d' %
          (len(train_data), len(dev_data), len(test_data)))

    print('Saving...')
    save('train-data', train_data)
    save('dev-data', dev_data)
    save('test-data', test_data)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Seperator')

    parser.add_argument(
        '--data',
        default="/Users/liangqiu/Workspace/#ubuntu_test/#ubuntu_test.gz",
        help='Data')

    argv = parser.parse_args()
    print
    print(argv)

    main(argv)
