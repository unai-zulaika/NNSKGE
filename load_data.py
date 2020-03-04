"""
from https://github.com/ibalazevic/TuckER
"""


class Data:
    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        self.entity_ids_to_readable = self.load_entity_ids_to_readable()

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        return data

    def load_entity_ids_to_readable(self):
        with open("dict_entities.txt", "r") as f:
            data = f.read().strip().split("\n")

            ent_dict = {}
            for i in data:
                split = i.split()
                ent_dict[split[0]] = split[1:]

        return ent_dict

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(
            list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities