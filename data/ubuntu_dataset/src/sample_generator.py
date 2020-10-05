import argparse
import os
from pathlib import Path

from utils.io_utils import say, load_dataset, output_samples, output_vocab
from utils.sample import Sample
from utils.stats import sample_statistics


def get_samples(threads, n_prev_sents, test=False):
    """
    :param threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :return: samples: 1D: n_samples; elem=Sample()
    """

    if threads is None:
        return None

    say('\n\n\tTHREADS: {:>5}'.format(len(threads)))

    samples = []
    max_n_agents = n_prev_sents + 1

    for thread in threads:
        samples += get_one_thread_samples(thread, max_n_agents, n_prev_sents,
                                          test)

    # sample_statistics(samples, max_n_agents)

    return samples


def get_one_thread_samples(thread, max_n_agents, n_prev_sents, test=False):
    samples = []
    sents = []
    # agents_in_ctx = set([])

    for i, sent in enumerate(thread):
        time = sent[0]
        spk_id = sent[1]
        adr_id = sent[2]
        label = sent[-1]

        if adr_id == '-':
            if len(sents) > 2:
                # TODO: hyperparameter
                if len(sents) > 9:
                    sents = sents[:9]
                sample_json = Sample(sents)
                samples.append(sample_json)
            sents = []
        else:
            responses = sent[3:-1]
            original_sent = get_original_sent(responses, label)
            sents.append((time, spk_id, adr_id, original_sent))

    return samples


def is_sample(context, spk_id, adr_id, agents_in_ctx):
    if context is None:
        return False
    if spk_id == adr_id:
        return False
    if adr_id not in agents_in_ctx:
        return False
    return True


def get_context(i, sents, n_prev_sents, label, test=False):
    # context: 1D: n_prev_sent, 2D: (time, speaker_id, addressee_id, tokens, label)
    context = None
    if label > -1:
        if len(sents) >= n_prev_sents:
            context = sents[i - n_prev_sents:i]
        elif test:
            context = sents[:i]
    return context


def get_original_sent(responses, label):
    if label > -1:
        return responses[label]
    return responses[0]


def limit_sent_length(sents, max_n_words):
    return [sent[:max_n_words] for sent in sents]


def get_datasets(argv):
    # dataset: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    say('\n\nLoad dataset...')
    train_dataset, word_set = load_dataset(fn=argv.train_data)
    dev_dataset, _ = load_dataset(fn=argv.dev_data)
    test_dataset, _ = load_dataset(fn=argv.test_data)
    return train_dataset, dev_dataset, test_dataset, word_set


def create_samples(argv, train_dataset, dev_dataset, test_dataset):
    n_prev_sents = argv.n_prev_sents

    # samples: 1D: n_samples; elem=Sample()
    say('\n\nCreating samples...')
    train_samples = get_samples(threads=train_dataset,
                                n_prev_sents=n_prev_sents)
    dev_samples = get_samples(threads=dev_dataset,
                              n_prev_sents=n_prev_sents,
                              test=True)
    test_samples = get_samples(threads=test_dataset,
                               n_prev_sents=n_prev_sents,
                               test=True)
    return train_samples, dev_samples, test_samples


def main(argv):
    say('\nSAMPLE GENERATOR\n')

    path = Path(argv.train_data)
    os.chdir(path.parent)

    train_dataset, dev_dataset, test_dataset, word_dict = get_datasets(argv)
    train_samples, dev_samples, test_samples = create_samples(
        argv, train_dataset, dev_dataset, test_dataset)

    # n_cands = len(train_samples[0].response)
    # n_prev_sents = argv.n_prev_sents

    output_samples('train-sample', train_samples)
    output_samples('dev-sample', dev_samples)
    output_samples('test-sample', test_samples)
    output_vocab(word_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample Generator')

    parser.add_argument(
        '--train_data',
        default='/Users/liangqiu/Workspace/#ubuntu_test/train-data.gz',
        help='path to data')
    parser.add_argument(
        '--dev_data',
        default='/Users/liangqiu/Workspace/#ubuntu_test/dev-data.gz',
        help='path to data')
    parser.add_argument(
        '--test_data',
        default='/Users/liangqiu/Workspace/#ubuntu_test/test-data.gz',
        help='path to data')

    parser.add_argument('--n_prev_sents',
                        type=int,
                        default=5,
                        help='prev sents')

    argv = parser.parse_args()
    print
    print(argv)

    main(argv)
