import numpy as np
import torch


def getDistRatio(embeddings_indexes, k=5):
    embeddings = embeddings_indexes.weight
    sorted, indexes = torch.sort(embeddings, dim=0, descending=True)

    topk = torch.index_select(sorted, 0, torch.arange(k).cuda())

    intruder_index = int(sorted.size()[0] - (sorted.size()[0] / 10))

    intruders = torch.index_select(
        sorted, 0,
        torch.tensor(intruder_index).cuda()).squeeze()

    return distRatio(topk, intruders, intruder_index, embeddings_indexes,
                     indexes)


def distRatio(topk,
              intruder,
              intruder_index,
              embeddings_indexes,
              indexes,
              k=5):

    d = embeddings_indexes.weight.shape[1]
    distRatio = [(interdist(topk[:, i], intruder[i], intruder_index,
                            embeddings_indexes, indexes[:, i]) /
                  intradist(topk[:, i], embeddings_indexes, indexes[:, i]))
                 for i in range(d)]
    return sum(distRatio) / d


def intradist(topk, embeddings_indexes, indexes, k=5):
    intradist = [(torch.dist(embeddings_indexes(indexes[i]),
                             embeddings_indexes(indexes[j])) / k * (k - 1))
                 for i, x in enumerate(topk) for j, y in enumerate(topk)
                 if i != j]

    return sum(intradist)


def interdist(topk,
              intruder,
              intruder_index,
              embeddings_indexes,
              indexes,
              k=5):

    interdist = [(torch.dist(
        embeddings_indexes(indexes[i]),
        embeddings_indexes(torch.LongTensor([intruder_index]).cuda())) / k)
                 for i, x in enumerate(topk)]
    return sum(interdist)
