import requests
import io
import os
from tqdm import tqdm
import urllib
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

from load_data import Data


def request_fb_id(fb_id, api_key='AIzaSyBBRq-8DyIaUtldtO1SLbftaXHAjlD_qxQ'):
    r = requests.get(
        'https://kgsearch.googleapis.com/v1/entities:search?ids={ID}&key={API_KEY}'
        .format(ID=urllib.parse.quote(fb_id, safe=''), API_KEY=api_key))

    json_r = r.json()

    try:
        json_r = json_r['itemListElement'][0]['result']['name']
    except (KeyError, IndexError):
        json_r = 'None'

    return json_r


def save_ids_to_file(ids_to_ents, filename='labels'):
    path = '{}.txt'.format(filename)

    if not os.path.exists(path):
        f = io.open(path, "w", encoding='utf8')

        for k in tqdm(ids_to_ents):
            human_label = request_fb_id(k)
            f.write('{}\t{}\n'.format(k, human_label))
        f.close()

def wn_get_entities(d, filename='dict_entities_wn'):
    path = '{}.txt'.format(filename)

    if not os.path.exists(path):
        f = io.open(path, "w", encoding='utf8')

        for w in d.entities:
            try: 
                f.write('{}\t{}\n'.format(int(w), wordnet.synset_from_pos_and_offset('n' , int(w)).lemma_names()))
            except:
                f.write('{}\tNone\n'.format(int(w)))

        f.close()

data_dir = "data/%s/" % 'WN18RR'
d = Data(data_dir=data_dir, reverse=True)
wn_get_entities(d)

        

# save_ids_to_file(d.relations, filename='dict_relations')
