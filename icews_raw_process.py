import os
import random
import shutil
from preprocess import Parser
from tools import FileTools
from tqdm import tqdm

old_version_path = os.path.abspath('./data/rawData/')
new_version_path = os.path.abspath('./data/standard/')

# os.chdir(old_version_path)
datasets = os.listdir(old_version_path)
print(datasets)

# def read_file_to_dict(file):
#     res_dict = {}
#     with open(file, 'r', encoding='utf-8') as rf:
#         for line in rf:
#             idx, value = line.strip().split('\t')
#             value = value.split("org/wiki/")[-1].replace("_", " ").replace("%", "").replace('\t', ' ').replace(u'\xa0', '')
#             res_dict[int(idx)] = value
#     # id_start = min(res_dict)
#     # res_dict = {idx-id_start: name for idx, name in res_dict.items()}
#     return res_dict

def read_file_to_dict(file):
    id_names = Parser.for_file(file, Parser.OEAFileType.ent_name)
    id_names_dict = {int(idx): value for idx, value in id_names}
    return id_names_dict

def load_oea_file(src, dst, filetype, long_short_name_map):
    tups = Parser.for_file(src, filetype)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            if long_short_name_map:
                tup = list(tup)
                tup[0] = long_short_name_map.get(tup[0], tup[0])
            print(*tup, sep='\t', file=wf)

def load_attr(src, dst):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)

def load_triple(src, dst, id_ent, id_rel):
    # id_ent = read_file_to_dict(f'ent_ids_{ds}')
    # id_rel = read_file_to_dict(f'rel_ids_{ds}')

    triples = []
    with open(src, 'r', encoding='utf-8') as rf:
        for line in rf:
            hid, rid, tid = [int(i) for i in line.strip().split('\t')[:3]]
            h, r, t = id_ent[hid], id_rel[rid], id_ent[tid]
            triples.append([h, r, t])
    
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in triples:
            print(*tup, sep='\t', file=wf)

def load_alignment(link_id_file):
    # def links_id_to_name(link_id_file, link_name_file):
    links = []
    id_ent_1 = read_file_to_dict(f'ent_ids_1')
    id_ent_2 = read_file_to_dict(f'ent_ids_2')
    with open(link_id_file, 'r', encoding='utf-8') as rf:
        # lines = rf.readlines()

    # with open(link_name_file, 'w', encoding='utf-8') as wf:
        for line in rf:
            sid, tid = line.strip().split('\t')
            src, tgt = id_ent_1[int(sid)], id_ent_2[int(tid)]
            links.append([src, tgt])
            # print(*(src, tgt), sep='\t', file=wf)
    # print('Saving links name to', link_name_file)
    return links

def change_id_to_start_0(id_file, new_id_file=None):
    idx_name_dict = read_file_to_dict(id_file)
    long_short_name_dict = {}
    if dataset in ['doremus_en', 'agrold_en']:  # 名字太长，替换成随机的短字符
        res_dict, long_short_name_dict = convert_long_name_short(idx_name_dict)

    id_start = min(res_dict)
    res_dict = {(idx-id_start): name for idx, name in res_dict.items()}
    
    if new_id_file:
        with open(new_id_file, 'w', encoding='utf-8') as fw:
            for tup in res_dict.items():
                print(*tup, sep='\t', file=fw)
        print('Saving new id_entity file to', new_id_file)
    return res_dict, long_short_name_dict

def convert_long_name_short(id_name_dict):
    import re
    import string
    # 定义匹配 UUID 的正则表达式模式
    uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
    def generate_random_string(idx):
        # 生成 E 加上 4 个随机数字的字符串
        # random_digits = ''.join(str(random.randint(0, 9)) for _ in range(4))
        return f'E{idx}'
    
    def replace_uuids(id, text):
        # 替换文本中的 UUID 为随机字符串
        def replace_with_random(match):
            return generate_random_string(id)
        return uuid_pattern.sub(replace_with_random, text)

    id_short_name_dict = {}
    long_short_name_dict = {}
    for id, long_name in id_name_dict.items():
        short_name = replace_uuids(id, long_name)
        id_short_name_dict[id] = short_name
        long_short_name_dict[long_name] = short_name
    return id_short_name_dict, long_short_name_dict

def links_map_new_name(links):
    for link in links:
        if link[0] in long_short_name_map1:
            link[0] = long_short_name_map1[link[0]]
        if link[1] in long_short_name_map2:
            link[1] = long_short_name_map2[link[1]]
    return links
import pickle

# 去掉三元组的链接
# datasets = ['icews_wiki']
datasets = ['doremus_en']
for dataset in datasets:
    os.chdir('/'.join((old_version_path, dataset)))
    # print(os.getcwd())
    # Save datasets whth new format to new path
    dataset_new_path = '/'.join((new_version_path, dataset+'_new'))
    if not os.path.exists(dataset_new_path):
        os.mkdir(dataset_new_path)

    # for icews_wiki, the idx of original target file does't begin with 0
    idx_ent1, long_short_name_map1 = change_id_to_start_0('ent_ids_1', '/'.join((dataset_new_path, 'id_entity_1.txt')))
    idx_ent2, long_short_name_map2 = change_id_to_start_0('ent_ids_2', '/'.join((dataset_new_path, 'id_entity_2.txt')))

    if dataset in ['doremus_en', 'agrold_en']:  # 名字太长，替换成随机的短字符
        # idx_ent1 = convert_long_name_short(idx_ent1)
        # idx_ent2 = convert_long_name_short(idx_ent2)
        
        ds1, ds2 = dataset.split('_')
        load_attr(f'{ds1}_att_triples', '/'.join((dataset_new_path, 'attr_triples_1')))
        load_oea_file('/'.join((dataset_new_path, 'attr_triples_1')), '/'.join((dataset_new_path, 'attr_triples_1')),
                    Parser.OEAFileType.attr, long_short_name_map1)
        load_attr(f'{ds2}_att_triples', '/'.join((dataset_new_path, 'attr_triples_2')))
        load_oea_file('/'.join((dataset_new_path, 'attr_triples_2')), '/'.join((dataset_new_path, 'attr_triples_2')),
                    Parser.OEAFileType.attr, long_short_name_map2)
        
    # transform to new format
    # ds1, ds2 = ['1', '2']
    idx_ent2 = {i+len(idx_ent1): name for i, name in idx_ent2.items()}  # 还原idx2从idx1之后开始，以保持和triple里一致
    id_rel1 = read_file_to_dict(f'rel_ids_1')
    id_rel2 = read_file_to_dict(f'rel_ids_2')
    load_triple(f'triples_1', dst='/'.join((dataset_new_path, 'rel_triples_1')), id_ent=idx_ent1, id_rel=id_rel1)
    load_triple(f'triples_2', dst='/'.join((dataset_new_path, 'rel_triples_2')), id_ent=idx_ent2, id_rel=id_rel2)

    # load_alignment(f'entity_seeds.txt', '/'.join((dataset_new_path, 'ent_links')), ds1, ds2)

    # ent_links = FileTools.load_list('/'.join((dataset_new_path, 'ent_links')))
    random.seed(11037)
    train_links = load_alignment(f'sup_pairs')
    valid_links = []
    test_links = load_alignment(f'ref_pairs')
    new_fold_path = '/'.join((new_version_path, dataset+'_new', '721_5fold', '1'))
    if not os.path.exists(new_fold_path):
        os.makedirs(new_fold_path)
    os.chdir(new_fold_path)

    if dataset in ['doremus_en', 'agrold_en']:
        train_links = links_map_new_name(train_links)
        valid_links = []
        test_links = links_map_new_name(test_links)

    FileTools.save_list(train_links, '/'.join((new_fold_path, 'train_links')))
    FileTools.save_list(valid_links, '/'.join((new_fold_path, 'valid_links')))
    FileTools.save_list(test_links, '/'.join((new_fold_path, 'test_links')))