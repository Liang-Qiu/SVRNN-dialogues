from ..utils.io_utils import say, load_dataset
from ..utils.sample import Sample
from ..utils.stats import sample_statistics, dataset_statistics


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
        samples += get_one_thread_samples(thread, max_n_agents, n_prev_sents, test)

    return samples


def get_one_thread_samples(thread, max_n_agents, n_prev_sents, test=False):
    samples = []
    sents = []
    agents_in_ctx = set([])

    for i, sent in enumerate(thread):
        time = sent[0]
        spk_id = sent[1]
        adr_id = sent[2]
        label = sent[-1]

        context = get_context(i, sents, n_prev_sents, label, test)
        responses = sent[3:-1]

        original_sent = get_original_sent(responses, label)
        sents.append((time, spk_id, adr_id, original_sent))

        agents_in_ctx.add(spk_id)

        ################################
        # Judge if it is sample or not #
        ################################
        if is_sample(context, spk_id, adr_id, agents_in_ctx):
            sample = Sample(context=context, spk_id=spk_id, adr_id=adr_id, responses=responses, label=label,
                            n_agents_in_ctx=len(agents_in_ctx), max_n_agents=max_n_agents)
            if test:
                samples.append(sample)
            else:
                # The num of the agents in the training samples is n_agents > 1
                # -1 means that the addressee does not appear in the limited context
                if sample.true_adr > -1:
                    samples.append(sample)

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
    train_samples = get_samples(threads=train_dataset, n_prev_sents=n_prev_sents)
    dev_samples = get_samples(threads=dev_dataset, n_prev_sents=n_prev_sents, test=True)
    test_samples = get_samples(threads=test_dataset, n_prev_sents=n_prev_sents, test=True)
    return train_samples, dev_samples, test_samples


def show_dataset_stats(train_dataset, dev_dataset, test_dataset):
    dataset_statistics(train_dataset)
    if dev_dataset is not None:
        dataset_statistics(dev_dataset)
    if test_dataset is not None:
        dataset_statistics(test_dataset)


def show_sample_stats(train_samples, dev_samples, test_samples, max_n_agents):
    sample_statistics(train_samples, max_n_agents)
    if dev_samples is not None:
        sample_statistics(dev_samples, max_n_agents)
    if test_samples is not None:
        sample_statistics(test_samples, max_n_agents)


def main(argv):
    say('\nSTATS VIEWER\n')

    train_dataset, dev_dataset, test_dataset, word_set = get_datasets(argv)
    show_dataset_stats(train_dataset, dev_dataset, test_dataset)
    train_samples, dev_samples, test_samples = create_samples(argv, train_dataset, dev_dataset, test_dataset)
    show_sample_stats(train_samples, dev_samples, test_samples, argv.n_prev_sents + 1)
