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


def load_oea_file(src, dst, filetype, old_new_name_dict):
    tups = Parser.for_file(src, filetype)
    if mask_name_ratio > 0:
        reduce_attr_ratio = mask_name_ratio
        random.shuffle(tups)
        num_reduced = int(len(tups) * reduce_attr_ratio)
        tups = tups[:-num_reduced]
        print('Reduced #attr:', num_reduced)
        
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            if old_new_name_dict:
                tup = list(tup)
                tup[0] = old_new_name_dict.get(tup[0], tup[0])
            print(*tup, sep='\t', file=wf)


def load_attr(src, dst):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)

def read_file_to_dict(file):
    res_dict = {}
    with open(file, 'r', encoding='utf-8') as rf:
        for line in rf:
            idx, value = line.strip().split('\t')
            res_dict[int(idx)] = value
    return res_dict

def load_triple(src, dst, id_ent, id_rel):
    triples = []
    with open(src, 'r', encoding='utf-8') as rf:
        for line in rf:
            hid, tid, rid = [int(i) for i in line.strip().split('\t')]
            h, r, t = id_ent[hid], id_rel[rid], id_ent[tid]
            triples.append([h, r, t])
    
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in triples:
            print(*tup, sep='\t', file=wf)

def load_alignment(link_id_file, link_name_file, ds1, ds2):
    id_ent_1 = read_file_to_dict(f'id2entity_{ds1}.txt')
    id_ent_2 = read_file_to_dict(f'id2entity_{ds2}.txt')

    links = []
    with open(link_id_file, 'r', encoding='utf-8') as rf:
        for line in rf:
            sid, tid = line.strip().split('\t')
            src, tgt = id_ent_1[int(sid)], id_ent_2[int(tid)]
            links.append([src, tgt])

    # keep 1-to-1 constraint, delete duplicate entities
    duplicate_pairs = []
    for i in range(len(links)):
        for j in range(i + 1, len(links)):
            if links[i][0] == links[j][0] or links[i][1] == links[j][0]:
                duplicate_pairs.append((links[i], links[j]))
    elements_to_remove = {tuple(item) for pair in duplicate_pairs for item in pair[:-1]} # keep one anchor pair
    links = [link for link in links if tuple(link) not in elements_to_remove]

    with open(link_name_file, 'w', encoding='utf-8') as wf:
        for tup in links:
            print(*tup, sep='\t', file=wf)

def change_name_to_random(id_name_dict, change_r=0.):
    id_name_list = list(id_name_dict.items())
    random.shuffle(id_name_list)
    change_count = int(len(id_name_list) * change_r)  # mask名字的实体数
    
    id_new_name_dict = id_name_dict.copy()
    old_new_name_dict = {name: name for idx, name in id_name_dict.items()}

    len_digit = len(str(len(id_name_list)))
    for idx, old_name in id_name_list[:change_count]:
        new_name ='E' + ''.join(str(random.randint(0, 9)) for _ in range(len_digit))
        id_new_name_dict[idx] = new_name
        old_new_name_dict[old_name] = new_name
    return id_new_name_dict, old_new_name_dict

def write_new_name(id_name_dict, new_id_file):
    with open(new_id_file, 'w', encoding='utf-8') as fw:
        for tup in id_name_dict.items():
            print(*tup, sep='\t', file=fw)
    print('Saving new id_entity file to', new_id_file)


def links_map_new_name(links):
    for link in links:
        if link[0] in old_new_name_dict1:
            link[0] = old_new_name_dict1[link[0]]
        if link[1] in old_new_name_dict2:
            link[1] = old_new_name_dict2[link[1]]
    return links

import pickle

# 去掉三元组的链接
datasets = ['zh_vi']
mask_name_ratio = 0.4
if mask_name_ratio > 0:
    new_version_path = os.path.abspath('./data/mask_name/')

for dataset in datasets:
    os.chdir('/'.join((old_version_path, dataset)))
    ds1, ds2 = dataset.split('_')
    # print(os.getcwd())
    # Save datasets whth new format to new path
    dataset_new_path = '/'.join((new_version_path, dataset))
    if mask_name_ratio > 0:
        dataset_new_path = os.path.join(dataset_new_path, f'mask_ratio_{mask_name_ratio}')
    if not os.path.exists(dataset_new_path):
        os.mkdir(dataset_new_path)

    # copy 'id2entity' file to destination path
    shutil.copy(f'id2entity_{ds1}.txt', '/'.join((dataset_new_path, f'id_entity_1.txt')))
    shutil.copy(f'id2entity_{ds2}.txt', '/'.join((dataset_new_path, f'id_entity_2.txt')))
    id_name_dict1 = read_file_to_dict('/'.join((dataset_new_path, f'id_entity_1.txt')))
    id_name_dict2 = read_file_to_dict('/'.join((dataset_new_path, f'id_entity_2.txt')))

    old_new_name_dict1, old_new_name_dict2 = {}, {}
    if mask_name_ratio > 0:
        id_name_dict1, old_new_name_dict1 = change_name_to_random(id_name_dict1, change_r=mask_name_ratio)
        id_name_dict2, old_new_name_dict2 = change_name_to_random(id_name_dict2, change_r=mask_name_ratio)
        write_new_name(id_name_dict1, '/'.join((dataset_new_path, f'id_entity_1.txt')))
        write_new_name(id_name_dict2, '/'.join((dataset_new_path, f'id_entity_2.txt')))
        
    # transform to new format
    load_oea_file(f'atts_properties_{ds1}.txt', '/'.join((dataset_new_path, 'attr_triples_1')),
                  Parser.OEAFileType.attr, old_new_name_dict1)
    load_oea_file(f'atts_properties_{ds2}.txt', '/'.join((dataset_new_path, 'attr_triples_2')),
                  Parser.OEAFileType.attr, old_new_name_dict2)

    id_rel1 = read_file_to_dict(f'id2relation_{ds1}.txt')
    id_rel2 = read_file_to_dict(f'id2relation_{ds2}.txt')
    load_triple(f'triples_{ds1}.txt', '/'.join((dataset_new_path, 'rel_triples_1')), id_name_dict1, id_rel1)
    load_triple(f'triples_{ds2}.txt', '/'.join((dataset_new_path, 'rel_triples_2')), id_name_dict2, id_rel2)

    load_alignment(f'entity_seeds.txt', '/'.join((dataset_new_path, 'ent_links')), ds1, ds2)

    ent_links = FileTools.load_list('/'.join((dataset_new_path, 'ent_links')))
    random.seed(11037)
    random.shuffle(ent_links)
    ent_len = len(ent_links)
    # train_len = ent_len * 4 // 15
    # valid_len = ent_len // 30
    train_len = int(ent_len * 0.2)
    valid_len = int(ent_len * 0.1)
    train_links = ent_links[:train_len]
    valid_links = ent_links[train_len: train_len + valid_len]
    test_links = ent_links[train_len + valid_len:]
    # new_fold_path = '/'.join((new_version_path, dataset, '721_5fold', '1'))
    new_fold_path = '/'.join((dataset_new_path, '721_5fold', '1'))
    if not os.path.exists(new_fold_path):
        os.makedirs(new_fold_path)
    os.chdir(new_fold_path)

    if mask_name_ratio > 0:
        train_links = links_map_new_name(train_links)
        valid_links = []
        test_links = links_map_new_name(test_links)

    FileTools.save_list(train_links, '/'.join((new_fold_path, 'train_links')))
    FileTools.save_list(valid_links, '/'.join((new_fold_path, 'valid_links')))
    FileTools.save_list(test_links, '/'.join((new_fold_path, 'test_links')))