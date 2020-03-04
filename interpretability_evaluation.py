import torch


def pick_top_k(embeddings_indexes, ent_dict, idxs_entity, k=5):
    embeddings = embeddings_indexes.weight
    _, indexes = torch.sort(embeddings, dim=0, descending=True)

    topk_words = [[0 for i in range(k + 1)]
                  for j in range(embeddings.shape[1])]

    # foreach dimension pick the top k embeddings
    for d in range(embeddings.shape[1]):
        for i in range(k):
            topk_words[d][i] = ' '.join(
                ent_dict[idxs_entity[indexes[i, d].item()]])
        topk_words[d][-1] = ' '.join(
            ent_dict[idxs_entity[indexes[-1, d].item()]])

    f = open("test.txt", "w+")
    for d in range(embeddings.shape[1]):
        for i in range(k + 1):
            f.write(topk_words[d][i] + ', ')
        f.write("\n")


def getDistRatio(embeddings_indexes, k=5):
    embeddings = embeddings_indexes.weight
    _, indexes = torch.sort(embeddings, dim=0, descending=True)

    topk = torch.index_select(sorted, 0, torch.arange(k).cuda())

    intruder_index = int(sorted.size()[0] - (sorted.size()[0] / 10))

    intruders = torch.index_select(
        sorted, 0,
        torch.tensor(intruder_index).cuda()).squeeze()

    return distRatio(topk, intruders, intruder_index, embeddings_indexes,
                     indexes)


def distRatio(topk, intruder, intruder_index, embeddings_indexes, indexes):

    d = embeddings_indexes.weight.shape[1]
    return sum([(interdist(topk[:, i], intruder[i], intruder_index,
                           embeddings_indexes, indexes[:, i]) /
                 intradist(topk[:, i], embeddings_indexes, indexes[:, i]))
                for i in range(d)]) / d


def intradist(topk, embeddings_indexes, indexes, k=5):
    return sum([(torch.dist(embeddings_indexes(indexes[i]),
                            embeddings_indexes(indexes[j])) / k * (k - 1))
                for i, x in enumerate(topk) for j, y in enumerate(topk)
                if i != j])


def interdist(topk, intruder, intruder_index, embeddings_indexes, indexes,
              k=5):

    return sum([(torch.dist(
        embeddings_indexes(indexes[i]),
        embeddings_indexes(torch.LongTensor([intruder_index]).cuda())) / k)
                for i, x in enumerate(topk)])
