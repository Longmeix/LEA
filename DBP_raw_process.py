import os
import random
import shutil
from preprocess import Parser
from tools import FileTools
from tqdm import tqdm

old_version_path = os.path.abspath('../DBP15k')
# old_version_path = os.path.abspath('./data/rawData/')
new_version_path = os.path.abspath('./data/standard')

os.chdir(old_version_path)
datasets = os.listdir('.')
print(datasets)


def load_oea_file(src, dst, filetype):
    tups = Parser.for_file(src, filetype)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            print(*tup, sep='\t', file=wf)


def load_attr(src, dst):
    tups = Parser.for_file(src, Parser.OEAFileType.ttl_full)
    with open(dst, 'w', encoding='utf-8') as wf:
        for tup in tups:
            if tup == 'Error':
                continue
            print(*tup, sep='\t', file=wf)

def relevant_triples(file, select_ents, type='attr'):
    tups = Parser.for_file(file, Parser.OEAFileType.rel)
    sel_tups = []
    if type == 'attr':
        for s, p, v in tups:
            if s in select_ents:
                sel_tups.append([s, p, v])
    elif type == 'rel':
        for s, p, o in tups:
            if s in select_ents and o in select_ents:
                sel_tups.append([s, p, o])

    with open(file, 'w', encoding='utf-8') as wf:
        for tup in sel_tups:
            print(*tup, sep='\t', file=wf)

# from preprocess.KBConfig import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, metavar='dataset path', default='D_W_15K')
args = parser.parse_args()

# # 去掉三元组的链接
datasets = ['zh_en']
# datasets = [args.dataset]
for dataset in datasets:
    os.chdir('/'.join((old_version_path, dataset)))

    # print(os.getcwd())
    # 复制数据集文件
    dataset_new_path = '/'.join((new_version_path, dataset))
    if not os.path.exists(dataset_new_path):
        os.mkdir(dataset_new_path)

    if dataset in ['zh_en', 'zh_vi', 'fr_en', 'ja_en']:
        ds1, ds2 = dataset.split('_')
        load_attr('_'.join((ds1, 'att_triples')), '/'.join((dataset_new_path, 'attr_triples_1')))
        load_oea_file('/'.join((dataset_new_path, 'attr_triples_1')), '/'.join((dataset_new_path, 'attr_triples_1')),
                    Parser.OEAFileType.attr)
        load_attr('_'.join((ds2, 'att_triples')), '/'.join((dataset_new_path, 'attr_triples_2')))
        load_oea_file('/'.join((dataset_new_path, 'attr_triples_2')), '/'.join((dataset_new_path, 'attr_triples_2')),
                    Parser.OEAFileType.attr)
        load_oea_file('_'.join((ds1, 'rel_triples')), '/'.join((dataset_new_path, 'rel_triples_1')), Parser.OEAFileType.rel)
        load_oea_file('_'.join((ds2, 'rel_triples')), '/'.join((dataset_new_path, 'rel_triples_2')), Parser.OEAFileType.rel)
        shutil.copy('ent_ILLs', '/'.join((dataset_new_path, 'ent_links')))
        # delete the redundant entities of DBP15K 
        if dataset in ['zh_en', 'fr_en', 'ja_en']:
            entity_1 = [name for i, name in Parser.for_file('_'.join((ds1, 'ent_ids')), Parser.OEAFileType.ent_name)]
            entity_2 = [name for i, name in Parser.for_file('_'.join((ds2, 'ent_ids')), Parser.OEAFileType.ent_name)]
            # relevant_triples('/'.join((dataset_new_path, 'attr_triples_1')), entity_1, 'attr')
            # relevant_triples('/'.join((dataset_new_path, 'attr_triples_2')), entity_2, 'attr')
            relevant_triples('/'.join((dataset_new_path, 'rel_triples_1')), entity_1, 'rel')
            relevant_triples('/'.join((dataset_new_path, 'rel_triples_2')), entity_2, 'rel')

    if dataset in ['D_W_15K', 'D_Y_15K']:
        load_attr('attr_triples_1', '/'.join((dataset_new_path, 'attr_triples_1')))
        load_oea_file('/'.join((dataset_new_path, 'attr_triples_1')), '/'.join((dataset_new_path, 'attr_triples_1')),
                    Parser.OEAFileType.attr)
        load_attr('attr_triples_2', '/'.join((dataset_new_path, 'attr_triples_2')))
        load_oea_file('/'.join((dataset_new_path, 'attr_triples_2')), '/'.join((dataset_new_path, 'attr_triples_2')),
                    Parser.OEAFileType.attr)
        load_oea_file('rel_triples_1', '/'.join((dataset_new_path, 'rel_triples_1')), Parser.OEAFileType.rel)
        load_oea_file('rel_triples_2', '/'.join((dataset_new_path, 'rel_triples_2')), Parser.OEAFileType.rel)
        shutil.copy('ent_links', '/'.join((dataset_new_path, 'ent_links')))
    
    load_oea_file('/'.join((dataset_new_path, 'ent_links')), '/'.join((dataset_new_path, 'ent_links')),
                  Parser.OEAFileType.truth)
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
    new_fold_path = '/'.join((new_version_path, dataset, '721_5fold', '1'))
    if not os.path.exists(new_fold_path):
        os.makedirs(new_fold_path)
    os.chdir(new_fold_path)

    FileTools.save_list(train_links, '/'.join((new_fold_path, 'train_links')))
    FileTools.save_list(valid_links, '/'.join((new_fold_path, 'valid_links')))
    FileTools.save_list(test_links, '/'.join((new_fold_path, 'test_links')))
    FileTools.save_list(ent_links, '/'.join((new_fold_path, 'ent_links')))


    # # # ===== 处理KG的attr变成一段文本 =====
    # print("Generating id for entities, translating KGs to text")
    # from preprocess.KBStore import KBStore
    # dataset1 = Dataset(1)
    # dataset2 = Dataset(2)
    # print('dataset1:', dataset1)
    # print('dataset2:', dataset2)

    # fs1 = KBStore(dataset1)
    # fs2 = KBStore(dataset2)
    # fs1.load_kb()
    # fs2.load_kb()
    # data_dir = os.path.dirname(dataset1.attr_seq_out)
    # concat_attr_relation_seq(dataset1.attr_seq_out, dataset1.neighboronly_seq_out, os.path.join(data_dir, 'attr_rel_seq_1.txt'))  # 将attr和relation产生的句子拼起来，更全面的信息
    # concat_attr_relation_seq(dataset2.attr_seq_out, dataset2.neighboronly_seq_out, os.path.join(data_dir, 'attr_rel_seq_2.txt'))
    # print('Done')