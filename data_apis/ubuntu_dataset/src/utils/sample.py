import numpy as np
import random


class Sample(object):
    def __init__(self, sents):

        self.ans_idx = random.randrange(1, len(sents))

        # 1D: n_prev_sents, 2D: max_n_words
        self.context = [s[-1] for s in sents]

        self.answer = self.context[self.ans_idx]
        del self.context[self.ans_idx]

        # important: ans_idx is the index of sentence which is followed by the answer in context list
        self.ans_idx -= 1

        self.relation_at = []
        self.relation_user = []

        for i, sent in enumerate(sents):
            spk_id = sent[1]
            adr_id = sent[2]

            for j, s in enumerate(sents[i + 1:]):
                if s[1] == adr_id and s[2] == spk_id:
                    self.relation_at.append([i + j + 1, i])
                    break

            for j, s in enumerate(sents[i + 1:]):
                if s[1] == spk_id:
                    self.relation_user.append([i + j + 1, i])


def get_adr_label(addressee_id, agent_index_dict):
    """
    :param addressee_id: the addressee of the response; int
    :param agent_index_dict: {agent id: agent index}
    """

    n_agents_lctx = len(agent_index_dict)

    # the case of including addressee in the limited context
    if addressee_id in agent_index_dict and n_agents_lctx > 1:
        # TODO: why minus 1?
        true_addressee = agent_index_dict[addressee_id] - 1
    else:
        true_addressee = -1

    return true_addressee


def get_adr_label_vec(adr_id, agent_index_dict, max_n_agents):
    """
    :param adr_id: the addressee of the response; int
    :param agent_index_dict: {agent id: agent index}
    """

    y = []
    n_agents_lctx = len(agent_index_dict)

    # the case of including addressee in the limited context
    if adr_id in agent_index_dict and n_agents_lctx > 1:
        # True addressee index
        y.append(agent_index_dict[adr_id] - 1)

        # False addressee index
        for i in range(len(agent_index_dict) - 1):
            if i not in y:
                y.append(i)

    pad = [-1 for i in range(max_n_agents - 1 - len(y))]
    y = y + pad
    return y


def get_false_res_label(response, label):
    """
    :param response: [response1, response2, ... ]
    :param label: true response label; int
    :return: int
    """
    n_responses = len(response)
    cand_indices = list(range(n_responses))
    cand_indices.remove(label)
    np.random.shuffle(cand_indices)
    return cand_indices


def get_spk_agent_one_hot_vec(context, agent_index_dict, max_n_agents):
    """
    :param context: 1D: n_prev_sents, 2D: n_words
    :param agent_index_dict: {agent id: agent index}
    :param max_n_agents: the max num of agents that appear in the context (=n_prev_sents+1); int
    :return: 1D: n_prev_sents, 2D: max_n_agents
    """
    speaking_agent_one_hot_vector = []
    for c in context:
        vec = [0 for i in range(max_n_agents)]
        speaker_id = c[1]
        vec[agent_index_dict[speaker_id]] = 1
        speaking_agent_one_hot_vector.append(vec)
    return speaking_agent_one_hot_vector


def indexing(responding_agent_id, context):
    agent_ids = {responding_agent_id: 0}
    for c in reversed(context):
        agent_id = c[1]
        if agent_id not in agent_ids:
            agent_ids[agent_id] = len(agent_ids)
    return agent_ids


def padding_response(responses, max_n_words):
    pads = []
    for sent in responses:
        diff = max_n_words - len(sent)
        pad = [0 for i in xrange(diff)]
        pad.extend(sent)
        pads.append(pad)
    return pads


def padding_context(context, max_n_words):
    def padding_sent(_sent):
        diff = max_n_words - len(_sent)
        return [0 for i in xrange(diff)] + _sent

    return [padding_sent(sent[-1]) for sent in context]


def bin_n_agents_in_ctx(n):
    if n < 6:
        return 0
    elif n < 11:
        return 1
    elif n < 16:
        return 2
    elif n < 21:
        return 3
    elif n < 31:
        return 4
    elif n < 101:
        return 5
    return 6
