import requests
import io
import os
from tqdm import tqdm
import urllib

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


data_dir = "data/%s/" % 'FB15k-237'
d = Data(data_dir=data_dir, reverse=True)

save_ids_to_file(d.relations, filename='dict_relations')
