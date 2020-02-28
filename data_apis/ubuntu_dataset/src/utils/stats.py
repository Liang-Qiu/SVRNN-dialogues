from collections import defaultdict

from .io_utils import say


def dataset_statistics(dataset):
    """
    :param dataset: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    """
    n_docs = len(dataset)
    n_utterances = 0
    n_words = 0
    n_agents = 0
    max_n_agents = 0

    for thread in dataset:
        agents = set([])
        n_utterances += len(thread)
        for sent in thread:
            label = sent[-1]
            if label > -1:
                sent_len = len(sent[3 + label])
            else:
                sent_len = len(sent[3])
            n_words += sent_len
            agents.add(sent[1])

        n_agents_tm = len(agents)
        n_agents += n_agents_tm
        if max_n_agents < n_agents_tm:
            max_n_agents = n_agents_tm

    say('\nDATASET STATS\n# Docs: {:>4} | # Utterances: {:>8} | # Words: {:>8}\n'
        .format(n_docs, n_utterances, n_words))
    say('# Agents: {:>8} | # Max agents/Doc: {:>3}\n'.format(
        n_agents, max_n_agents))
    say('Words/Utter: {:3.2f} | Agents/Doc: {:3.2f}\n'.format(
        n_words / float(n_utterances), n_agents / float(n_docs)))


def sample_statistics(samples, max_n_agents):
    show_adr_chance_level(samples)
    show_adr_upper_bound(samples, max_n_agents)
    show_n_samples_binned_ctx(samples)


def show_adr_chance_level(samples):
    total = float(len(samples))
    total_agents = 0.
    stats = defaultdict(int)

    for sample in samples:
        stats[sample.n_agents_in_ctx] += 1

    for n_agents, n_samples in stats.items():
        assert n_agents > 0
        total_agents += n_agents * n_samples

    say('\n\t  SAMPLES: {:>8}'.format(int(total)))
    say('\n\t  ADDRESSEE DETECTION CHANCE LEVEL: {:>7.2%}'.format(
        total / total_agents))


def show_adr_upper_bound(samples, max_n_agents):
    true_adr_stats = defaultdict(int)
    non_adr_stats = defaultdict(int)

    # sample.n_agents_in_lctx = agents appearing in the limited context (including the speaker of the response)
    for sample in samples:
        if sample.true_adr > -1:
            true_adr_stats[sample.n_agents_in_lctx] += 1
        else:
            non_adr_stats[sample.n_agents_in_lctx] += 1

    say('\n\t  ADDRESSEE DETECTION UPPER BOUND:')
    for n_agents in range(max_n_agents):
        n_agents += 1
        if n_agents in true_adr_stats:
            ttl1 = true_adr_stats[n_agents]
        else:
            ttl1 = 0
        if n_agents in non_adr_stats:
            ttl2 = non_adr_stats[n_agents]
        else:
            ttl2 = 0
        total = float(ttl1 + ttl2)

        if total == 0:
            ub = 0.
        else:
            ub = ttl1 / total

        say('\n\t\t# Cands {:>2}: {:>7.2%} | Total: {:>8} | Including true-adr: {:>8} | Not including: {:>8}'
            .format(n_agents, ub, int(total), ttl1, ttl2))
    say('\n')


def show_n_samples_binned_ctx(samples):
    ctx_stats = defaultdict(int)
    for sample in samples:
        ctx_stats[sample.binned_n_agents_in_ctx] += 1

    say('\n\t  THE BINNED NUMBER OF AGENTS IN CONTEXT:')
    for n_agents, ttl in sorted(ctx_stats.items(), key=lambda x: x[0]):
        say('\n\t\tBin {:>2}: {:>8}'.format(n_agents, ttl))
    say('\n')
