# import os
# os.environ['http_proxy'] = "172.18.166.31:7899"
# os.environ['https_proxy'] = os.environ['http_proxy']

import argparse
import datetime
import os
import random
import sys
import torch as t
import numpy as np
# from transformers import BertConfig, BertTokenizer, AutoTokenizer

from tools import Logger

seed = 11037
random.seed(seed)
t.manual_seed(seed)
t.cuda.manual_seed_all(seed)
np.random.seed(seed)

time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')[:-3]
version_str = time_str[:10]
run_file_name = sys.argv[0].split('/')[-1].split('.')[0]

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='OpenEA')
# parser.add_argument('--mode', type=str, default='KB') # filter some unimportant attribute
parser.add_argument('--log', action='store_true', default=False)
# ================= Dataset ===============================================
data_group = parser.add_argument_group(title='General Dataset Options')
data_group.add_argument('--datasets_root', type=str, default='./data/standard')
data_group.add_argument('--result_root', type=str, default='output')
data_group.add_argument('--functionality', action='store_true') # filter some unimportant attribute
data_group.add_argument('--blocking', action='store_true')
data_group.add_argument('--relation', action='store_true', default=True)
data_group.add_argument('--attribute', action='store_true', default=True)
# =========================================================================
# ================= OpenEA ================================================
openea_group = parser.add_argument_group(title='OpenEA Dataset Options')
openea_group.add_argument('--dataset', type=str, metavar='dataset path', default='doremus_en')
openea_group.add_argument('--fold', type=int, default=1)
# =========================================================================
train_group = parser.add_argument_group(title='Train Options')
train_group.add_argument('--gpus', type=str, default='1')
train_group.add_argument('--version', type=str, default='{version}')
# ================== LLM ====================
llm_group = parser.add_argument_group(title='call LLM')
llm_group = parser.add_argument('--llm', type=str, default='deepseek', choices=['llama_8b', 'llama_70b', 'deepseek', 'gpt4omini'])
llm_group = parser.add_argument('--input_info', type=str, default='attr_rel', choices=['attr_rel', 'attr', 'rel', 'name', 'attr_ngb'])
llm_group = parser.add_argument('--answer_english', action='store_true', help="LLM answers in English")
llm_group = parser.add_argument('--instruct', type=str, default='normal', choices=['normal', 'short', 'simple'], help="instruction")
llm_group = parser.add_argument('--max_threads', type=int, default=1)


args = parser.parse_args()
# print("english: ", args.answer_english)
seq_max_len = 128
bert_output_dim = 300
PARALLEL = True
DEBUG = False
SCORE_DISTANCE_LEVEL, MARGIN = 2, 1
if args.version is not None:
    version_str = args.version
if args.relation:
    # version_str += '-relation'
    version_str = 'text_sequence'
args.attribute = False if  args.dataset in ["icews_wiki"] else args.attribute

# ================= OpenEA ================================================
dataset_name = args.dataset
dataset_home = os.path.join(args.datasets_root, dataset_name)
add_obj_ent = False if dataset_name in ['zh_en', 'fr_en', 'ja_en'] else True  #数据集有太多冗余目标实体，出度0入度1时可看作属性

ds = dataset_name.split('_')
# result_name = '-'.join((time_str, dataset_name, str(args.fold), run_file_name, str(seq_max_len)))
log_name = '-'.join((time_str, dataset_name, run_file_name, str(seq_max_len)))
result_home = '/'.join((args.datasets_root, dataset_name, args.result_root))
if not os.path.exists(result_home):
    os.makedirs(result_home)
need_log = args.log
log_path = os.path.join(result_home, 'logs')
if need_log:
    Logger.make_print_to_file(name=log_name + '-add.txt', path=log_path)
print("log_path", log_path)
print("log_name", log_name)


class Dataset:
    def __init__(self, no):
        self.dataset_home = dataset_home
        self.result_home = result_home
        self.name = ds[no - 1]
        self.attr = self.triples('attr', no)
        self.rel = self.triples('rel', no)
        self.entities_out = self.outputs_root_txt('id_entity', no)
        # self.literals_out = self.outputs_tab('literals', no)
        self.properties_out = self.outputs_res_txt('properties', no)
        self.relations_out = self.outputs_python('relations', no)
        # ----------------------- TEA --------------------------------------
        self.attr_seq_out = self.outputs_res_txt('attr_seq', no)
        self.neighboronly_seq_out = self.outputs_res_txt('rel_seq', no)
        self.attr_rel_seq_out = self.outputs_res_txt('attr_ngb_seq', no)  # attributes和邻居组成的实体信息
        # ----------------------------------------------------------------

    def __str__(self):
        return 'Dataset{name: %s, rel: %s, attr: %s}' % (self.name, self.rel, self.attr)

    @staticmethod
    def triples(name, no):
        file_name = '_'.join((name, 'triples', str(no)))
        return os.path.join(dataset_home, file_name)

    @staticmethod
    def outputs_tab(name, no):
        file_name = '_'.join((name, 'tab', str(no))) + '.txt'
        return os.path.join(result_home, file_name)
    
    @staticmethod
    def outputs_res_txt(name, no):
        file_name = '_'.join((name, str(no))) + '.txt'
        return os.path.join(result_home, file_name)
    
    @staticmethod
    def outputs_root_txt(name, no):
        file_name = '_'.join((name, str(no))) + '.txt'
        return os.path.join(dataset_home, file_name)

    @staticmethod
    def outputs_csv(name, no):
        file_name = '_'.join((name, 'csv', str(no))) + '.csv'
        return os.path.join(result_home, file_name)

    @staticmethod
    def outputs_python(name, no):
        file_name = '_'.join((name, 'python', str(no))) + '.txt'
        return os.path.join(result_home, file_name)

if args.mode == 'KB':
    functionality_control = True
else:
    functionality_control = args.functionality
print("==== Filter Attribute ====>", functionality_control)
functionality_threshold = 0.9

print('time str:', time_str)
print('run file:', run_file_name)
print('args:')
print(args)
print('log path:', os.path.abspath(log_path))
print('log:', need_log)

print('result_path:', result_home)

# ========================================================