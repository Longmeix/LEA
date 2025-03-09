# 编码为嵌入，再对齐，在一个文件中完成

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.chdir('../')

import torch
from tqdm import tqdm
import numpy as np
from transformers import logging
logging.set_verbosity_warning()
# from tools import FileTools
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
# import concurrent.futures as cf
import time
import argparse
from text_process import config as cfg
from alignment import eval_alignment_by_sim_mat

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['http_proxy'] = "172.18.166.31:7899"
os.environ['https_proxy'] = os.environ['http_proxy']

def bert_cls_embed(sentences, batch_size=256):
    cls_vectors = []

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_texts = sentences[i:i + batch_size]
        
        # 将输入传递给模型
        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors='pt', max_length=512, padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            # batch_cls_vectors = outputs.last_hidden_state[:, 0, :].cpu()  # cls向量
            batch_cls_vectors = outputs.last_hidden_state.mean(dim=1).cpu()  # 平均池化
            cls_vectors.append(batch_cls_vectors)
        del inputs

    cls_vectors = torch.cat(cls_vectors, dim=0)
    return cls_vectors.detach().numpy()

def encode_file_text(model, seqs, encoder_name):
    """
    读取文本文件，生成所有句子的嵌入，并按序号返回嵌入
    """
    if isinstance(seqs, str) and os.path.isfile(seqs):
        with open(seqs, 'r', encoding='utf-8') as file:
            seqs = [line.split('\t')[-1] for line in file.read().splitlines()]
    elif hasattr(seqs, 'read'):  # 判断 seqs 是否为文件对象
        seqs = [line.split('\t')[-1] for line in seqs.read().splitlines()]
    elif isinstance(seqs, list):
        pass
    else:
        raise ValueError("seqs 必须是文件名、文件对象或列表")
    
    if encoder_name in ['LABSE', 'BGEm3', 'ALLv2', 'sBERT']:
        embeddings = model.encode(seqs, normalize_embeddings=True, batch_size=args.encode_bsz)
    elif encoder_name.startswith("JINA"):
        embeddings = model.encode(seqs, task="text-matching", batch_size=args.encode_bsz)
    elif encoder_name in ['BERT', 'roberta']:
        embeddings = bert_cls_embed(seqs, batch_size=args.encode_bsz)
    else:
        raise ValueError("Please modify the code to add new encoder.")
    
    return embeddings.astype(np.float32)


def encode_two_kgs_test_links(text_file, save_emb_file):
    # ===== 将由attr/rel/attr+rel组成的句子 编码为 嵌入，作为文本嵌入 =====
    start_time = time.time()
    embs_all = []   
    for net_id in ('1', '2'):
        file_name = text_file.replace('seq', f'seq_{net_id}')
        seqs = []  # sentences to encode
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                seqs.append(line.split('\t')[-1])  # encode 最后的句子
        # 生成嵌入
        print(f'No embedding file {save_emb_file}.\n \
                Start encoding {file_name}...')
        embs = encode_file_text(model, seqs, encoder_name)
        embs_all.append(embs)
    assert len(embs_all) == 2 and embs_all[0].shape[0] == embs_all[1].shape[0]
    embs_all = np.vstack(embs_all)
    # save embeddings
    np.save(save_emb_file, embs_all)
    print('Save embedding to ', save_emb_file)
    print('Encoding time: {:.4f}\n'.format(time.time()-start_time))
    return embs_all


def extract_entities_from_file(filename, id_start=0):  
    """
    :param filename: 文件名
    :return: 字典，键为实体名字，值为实体id
    """
    entities = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split('\t')
            line_number, entity_name = line[0], line[1].strip()
            entities[entity_name] = int(line_number)+id_start
    return entities


def load_ent_links(ent_links_file):
    # load test anchor links
    ent_name_links = []
    with open(ent_links_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            ea_name_1, ea_name_2 = line.strip().split('\t')
            ent_name_links.append((ea_name_1, ea_name_2))

    ent_name_links = np.array(ent_name_links)
    return ent_name_links


def links_name2id(ent_name_links, ename_id_1, ename_id_2):
    ent_links_id = []
    for ea_name_1, ea_name_2 in ent_name_links:
        eid_1, eid_2 = ename_id_1[ea_name_1], ename_id_2[ea_name_2]
        ent_links_id.append((eid_1, eid_2))
    ent_links_id = np.array(ent_links_id)
    return ent_links_id


def reduce_dim_by_SVD(emb1, emb2, reduce_dim=256):
    entities_emb = np.vstack([emb1, emb2])
    U, S, _ = np.linalg.svd(entities_emb, full_matrices=False)
    entities_emb = U[:, :reduce_dim] @ np.diag(S[:reduce_dim])
    entities_emb = entities_emb / np.linalg.norm(entities_emb, axis=-1, keepdims=True)
    num_link = len(entities_emb) // 2
    emb1, emb2 = entities_emb[:num_link], entities_emb[num_link:]
    return emb1, emb2

def load_link_embs(embs_file):
    entities_emb = np.load(embs_file) 
    num_link = len(entities_emb) // 2
    emb1, emb2 = entities_emb[:num_link], entities_emb[num_link:]
    emb1, emb2 = reduce_dim_by_SVD(emb1, emb2)
    return emb1, emb2

def replace_origin_with_enhanced_texts(anchors, hard_ent_names, seq_file, llm_file):
    '''根据困难实体，找到原文本，然后用LLM增强'''
    # 读取seq_file
    seq_dict = {}
    with open(seq_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            e_name = line.strip().split('\t')[1]
            seq_dict[e_name] = line
    # 读取llm_file
    llm_dict = {}
    with open(llm_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            e_name = line.strip().split('\t')[1]
            llm_dict[e_name] = line
    # 遍历anchors，困难实体用LLM增强的文本替换原文本，并写入新文件
    cnt = 0
    dir_name, file_name = os.path.split(seq_file)
    new_dir = os.path.join(dir_name, 'refine_mix_llm')
    os.makedirs(new_dir, exist_ok=True)
    save_mix_llm_file = os.path.join(new_dir, file_name.replace('seq_', f'{llm}_hard_seq_'))

    with open(save_mix_llm_file, 'w', encoding='utf-8') as fw:
        for ea_name in anchors:
            if ea_name in hard_ent_names:
                combined_line = seq_dict[ea_name].rstrip() + ' ' + llm_dict[ea_name].split('\t')[-1]  #LLM文本拼接原文本
                fw.write(combined_line)
                cnt += 1
            else:
                fw.write(seq_dict[ea_name])
            # 用LLM替换原文本
            # if ea_name in hard_ent_names:
            #     fw.write(llm_dict[ea_name])
            #     cnt += 1
            # else:
            #     fw.write(seq_dict[ea_name])
    
    print('# hard: ', cnt)
    return save_mix_llm_file


def matching_difficulty(cand_sims):
    # 计算top[0]与top[-1]的差异项
    d_12 = cand_sims[-1]-cand_sims[0]
    # # 计算相邻差异的加权和
    weights = [1 / (i + 1) for i in range(len(cand_sims) - 1)]  # 权重递减
    D_adj = sum(w * abs(cand_sims[i] - cand_sims[i+1]) for i, w in enumerate(weights))
    # 设定权重系数
    alpha, beta = 10, 0.5
    # 计算匹配难度指标
    difficulty = d_12 + alpha * (1 / (1 + D_adj))
    return difficulty


def save_get_candidates(ent_name_links, Lvec, Rvec, eid_name_dict, cand_num=20, num_thread=16):
    from alignment import generate_candidates_by_sim_mat
    import json
    candidates = generate_candidates_by_sim_mat(Lvec, Rvec, ent_name_links, cand_num, csls=0, num_thread=num_thread)
    
    # 把candidates中的id更换为实体名，方便分析
    cands_str = {}
    cnt = 0
    match_difficulties = []
    for eid_1, cand_list in candidates.items():
        difficulty = matching_difficulty(cand_list['cand_sims'])
        match_difficulties.append(difficulty)
        cand_list['difficulty'] = difficulty
        if cand_list['ground_rank'] < 1:
            cnt += 1
        if cand_list['ground_rank'] > -1:
            cand_list['ref'] = eid_name_dict[cand_list['ref']]
            for i, cid in enumerate(cand_list['candidates']):
                cand_list['candidates'][i] = eid_name_dict[cid]
            cands_str[eid_name_dict[eid_1]] = cand_list
        
    # print('# top1: ', cnt, len(candidates))
    cands_str = dict(sorted(cands_str.items(), key=lambda item: item[1]['difficulty']))  # 难度从小到大排序

    # 存到文件
    saved_cand_path = os.path.join(args.datasets_root, args.dataset, "cand_LEA")
    if not os.path.exists(saved_cand_path):
        os.makedirs(saved_cand_path)
    with open(os.path.join(saved_cand_path, "cand_LEA"), "w", encoding="utf-8") as fw:
        json.dump(cands_str, fw, ensure_ascii=False, indent=4)
        print(f'Saved candidates (for ChatEA) to {os.path.join(saved_cand_path, "cand_LEA")}')

def get_hard_entities(test_alignment, Lvec, Rvec, eid_name_dict, replace_ratio, hard2easy='True', cand_num=20, num_thread=16):
    from alignment import generate_candidates_by_sim_mat
    import json
    candidates = generate_candidates_by_sim_mat(Lvec, Rvec, test_alignment, cand_num, csls=0, num_thread=num_thread)
    
    # 把candidates中的id更换为实体名，方便分析
    cands_str = {}
    cnt = 0
    match_difficulties = []
    for eid_1, cand_list in candidates.items():
        difficulty = matching_difficulty(cand_list['cand_sims'])
        match_difficulties.append(difficulty)
        cand_list['difficulty'] = difficulty
        cands_str[eid_name_dict[eid_1]] = cand_list


    # 按难度分数排序
    if hard2easy in ['True', 'False']:
        hard2easy = True if hard2easy == 'True' else False
        cands_str = dict(sorted(cands_str.items(), key=lambda item: item[1]['difficulty'], reverse=hard2easy))  # 从大到小
    elif hard2easy == 'Random':
        # 打乱字典
        import random
        random.seed(42)
        cands_str_items = list(cands_str.items())
        random.shuffle(cands_str_items)
        cands_str = dict(cands_str_items)
    else:
        raise ValueError("Please modify the code to add new method.")

    # 计算top 20% 的实体数量
    top_20_percent = int(len(cands_str) * replace_ratio)
    # 获取难度分数最大的top 20% 的实体
    hard_names = list(cands_str.keys())[:top_20_percent]

    # # 存到文件
    # saved_cand_path = os.path.join(args.datasets_root, args.dataset, "cand_LEA")
    # if not os.path.exists(saved_cand_path):
    #     os.makedirs(saved_cand_path)
    # with open(os.path.join(saved_cand_path, "cand_LEA"), "w", encoding="utf-8") as fw:
    #     json.dump(cands_str, fw, ensure_ascii=False, indent=4)
    #     print(f'Saved candidates (for ChatEA) to {os.path.join(saved_cand_path, "cand_LEA")}')

    return hard_names

def get_hard_tail_entities(anchors, seq_file, replace_ratio=0.2):
    # 读取seq_file
    ent_seq_len = {}
    # seq_dict = {}
    with open(seq_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            _, e_name, seq  = line.strip().split('\t')
            # seq_dict[e_name] = line
            if e_name in anchors:
                ent_seq_len[e_name] = len(seq)
    # 按实体原文本长度小到大排序实体
    ent_seq_len = dict(sorted(ent_seq_len.items(), key=lambda item: item[1]))
    # 取前xx个尾实体（文本少）
    num_top_20 = int(len(anchors) * replace_ratio)
    hard_ents = list(ent_seq_len.keys())[:num_top_20]
    print(f"# hard entities: {len(hard_ents)}")
    return hard_ents


def get_and_anc_seq_text_and_encode(anc_names, seq_file):
    ent_seqs = []
    ent_seq_dict = {}
    with open(seq_file, 'r') as fr:
        for line in fr:
            idx, ename = line.strip('\n').split('\t', maxsplit=2)[:2]
            ent_seq_dict[ename] = line
            # if ename in anc_names:
            #     ent_seqs.append(line.split('\t')[-1])

    ent_seqs = [ent_seq_dict[e].split('\t')[-1] for e in anc_names]
    emb = encode_file_text(model, ent_seqs, encoder_name)

    return emb

def get_and_save_anc_seq_text(e_names, seq_file, save_match_file):
    ent_seq_dict = {}
    with open(seq_file, 'r') as fr:
        for line in fr:
            idx, ename = line.strip('\n').split('\t', maxsplit=2)[:2]
            ent_seq_dict[ename] = line

    find_seq_dict = [ent_seq_dict.get(e, '') for e in e_names]
    # save to file
    with open(save_match_file, 'w', encoding='utf-8') as fw:
        fw.writelines(find_seq_dict)
    return find_seq_dict

    # 读取llm_file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encoding texts into embeddings")
    parser.add_argument("--datasets_root", default="./data/standard", type=str,help="dataset dir")
    parser.add_argument("--dataset", default="icews_wiki", type=str, help="dataset name")
    parser.add_argument("--llm", default="llama_70b", type=str, choices=["no_llm", "llama_8b", "llama_70b", "deepseek", "gpt4omini"], help="llm for enriching entity in 1th stage")
    parser.add_argument("--encoder", default="BERT", type=str, choices=["BERT", "roberta", "BGEm3", "sBERT", "JINAv3"], help="text encoder name in this(2rd) stage")
    parser.add_argument("--input_info", default="rel", type=str, help="what text to encode", 
                        choices=["name", "attr", "rel", "attr_rel", "attr_ngb"])  # attr_rel是直接将两类信息拼接起来，attr_ngb是有选择性的组合到一起
    parser.add_argument("--replace_ratio", default=1, type=float, help="ratio of entities in the original text replaced by LLM")
    parser.add_argument('--instruct', type=str, default='normal', choices=['normal', 'short', 'simple'], help="instruction")
    parser.add_argument("--encode_bsz", default=256, type=int, help="batch size of encoding")
    parser.add_argument("--device", default='cuda:0', type=str, help="cuda or gpu")
    parser.add_argument("--CSLS_k", default=0, type=int, help="for csls")
    args = parser.parse_args()
    cfg.init_args(args)
    print(args)

    # hyperparameters
    data_dir = cfg.data_dir
    device = args.device
    llm = args.llm.lower()
    encoder_name = args.encoder
    encoder = {'ALLv2': 'sentence-transformers/all-MiniLM-L6-v2',
                'roberta': 'FacebookAI/xlm-roberta-large',
                'BERT': 'google-bert/bert-base-multilingual-uncased',
                'LABSE': 'sentence-transformers/LaBSE',
                'sBERT': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'BGEm3': 'BAAI/bge-m3', 
                'JINAv3': 'jinaai/jina-embeddings-v3'
                }
    
    # 加载Encoder模型
    if encoder_name in ['LABSE', 'BGEm3', 'ALLv2', 'sBERT']:
        model = SentenceTransformer(encoder[encoder_name], device=device)
    elif encoder_name.startswith("JINA"):
        model = SentenceTransformer(encoder[encoder_name], trust_remote_code=True, device=device)
    elif encoder_name in ['BERT', 'roberta']:
        tokenizer = AutoTokenizer.from_pretrained(encoder[encoder_name])
        model = AutoModel.from_pretrained(encoder[encoder_name]).to(device)
    else:
        raise ValueError("Please modify the code to add new encoder.")
    
    ent_links_file = os.path.join(args.datasets_root, args.dataset, '721_5fold/1/test_links')
    ent_name_links = load_ent_links(ent_links_file)
    
    # # 编码名字
    # seq_name_dir = os.path.join(args.datasets_root, args.dataset)
    # ori_seq_file1 = os.path.join(seq_name_dir, f'id_entity_1.txt')
    # ori_seq_file2 = os.path.join(seq_name_dir, f'id_entity_2.txt')
    # ent_links_file = os.path.join(seq_name_dir, '721_5fold/1/test_links')
    # ent_name_links = load_ent_links(ent_links_file)
    # emb1 = get_and_anc_seq_text_and_encode(ent_name_links[:, 0].tolist(), ori_seq_file1)
    # emb2 = get_and_anc_seq_text_and_encode(ent_name_links[:, 1].tolist(), ori_seq_file2)
    # emb1, emb2 = reduce_dim_by_SVD(emb1, emb2, reduce_dim=256)
    # print(emb1.shape, emb2.shape)
    # eval_alignment_by_sim_mat(emb1, emb2, csls_k=args.CSLS_k)

    # ========= LLM增强后的性能 ==========
    llm_fname = f'{llm}_sort_{args.input_info}_seq_test.txt'
    llm_fname = f'{args.instruct}_' + llm_fname if args.instruct != 'normal' else llm_fname
    llm_file = os.path.join(data_dir, llm_fname)  #
    if not os.path.exists(llm_file):
        ancs_1, ancs_2 = ent_name_links[:, 0].tolist(), ent_name_links[:, 1].tolist()
        for net, anc in zip(["1", "2"], [ancs_1, ancs_2]):
            # 原文本句子
            llm_ori_file = os.path.join(data_dir, f'{llm}_sort_{args.input_info}_seq_{net}.txt')
            # 读取测试集（候选实体）的原文本句子
            save_test_seq_file = os.path.join(llm_ori_file.replace('.txt', '_test.txt'))
            anc_seq = get_and_save_anc_seq_text(anc, llm_ori_file, save_test_seq_file)

    embs_file = os.path.join(data_dir, f'emb_{llm}_{args.input_info}_seq_test_{encoder_name}.npy')  #
    encode_two_kgs_test_links(llm_file, save_emb_file=embs_file)
    emb1, emb2 = load_link_embs(embs_file)
    eval_alignment_by_sim_mat(emb1, emb2, csls_k=args.CSLS_k)

    # 原文本的性能
    seq_nollm_dir = os.path.join(args.datasets_root, args.dataset, 'output')
    ori_seq_file = os.path.join(seq_nollm_dir, f'{args.input_info}_seq_test.txt')
    embs_file = os.path.join(seq_nollm_dir, f'emb_{args.input_info}_seq_test_{encoder_name}.npy')
    # encode_two_kgs_test_links(ori_seq_file, save_emb_file=embs_file)
    emb1, emb2 = load_link_embs(embs_file)
    eval_alignment_by_sim_mat(emb1, emb2, csls_k=args.CSLS_k)

    # # 实验评估，查看未能成功匹配的anchors特征
    # ent_links_file = os.path.join(args.datasets_root, args.dataset, '721_5fold/1/test_links')
    # ent_name_links = load_ent_links(ent_links_file)
    ename_id_1 = extract_entities_from_file(os.path.join(args.datasets_root, args.dataset, 'id_entity_1.txt'))
    ename_id_2 = extract_entities_from_file(os.path.join(args.datasets_root, args.dataset, 'id_entity_2.txt'), id_start=len(ename_id_1))
    eid_name_dict = {eid: e_name for e_name, eid in ename_id_1.items()}
    eid_name_dict.update({eid: e_name for e_name, eid in ename_id_2.items()})
    # save_get_candidates(ent_name_links, emb1, emb2, ename_id_1, ename_id_2, cand_num=10, num_thread=16)  # 保存candidates, 便于分析


    # ----- 只用LLM处理部分困难实体, 简单实体用原文本 -----
    links_st = links_name2id(ent_name_links, ename_id_1, ename_id_2)
    # 找到源图匹配到目标图的困难实体
    hard_ent_name_1 = get_hard_entities(links_st, emb1, emb2, eid_name_dict, args.replace_ratio, cand_num=10, num_thread=16)
    # 交换第0列和第1列, 目标图匹配到源图的困难实体
    links_ts = links_st.copy()
    links_ts[:, [0, 1]] = links_st[:, [1, 0]]
    hard_ent_name_2 = get_hard_entities(links_ts, emb2, emb1, eid_name_dict, args.replace_ratio, cand_num=10, num_thread=16)
    # 困难实体用自定义的指标选，用LLM生成的文本替换原文本
    seq_mix_dir = os.path.join(args.datasets_root, args.dataset, 'output', "refine_mix_llm")
    seq_file_1 = ori_seq_file.replace('seq', 'seq_1')
    seq_file_2 = ori_seq_file.replace('seq', 'seq_2')
    llm_file_1 = llm_file.replace('seq', 'seq_1')
    llm_file_2 = llm_file.replace('seq', 'seq_2')
    print('LLM file: ', llm_file_1, llm_file_2)
    seq_llm_file_1 = replace_origin_with_enhanced_texts(ent_name_links[:, 0].tolist(), hard_ent_name_1, seq_file_1, llm_file_1)
    seq_llm_file_2 = replace_origin_with_enhanced_texts(ent_name_links[:, 1].tolist(), hard_ent_name_2, seq_file_2, llm_file_2)

    # 编码新文本，由原文本和LLM部分增强组成
    emb1 = encode_file_text(model, seq_llm_file_1, encoder_name)
    emb2 = encode_file_text(model, seq_llm_file_2, encoder_name)
    emb1, emb2 = reduce_dim_by_SVD(emb1, emb2, reduce_dim=256)
    print('---- Replace Hard ----')
    eval_alignment_by_sim_mat(emb1, emb2, csls_k=args.CSLS_k)

    # # 消融：随机选困难实体
    # hard_ent_name_1 = get_hard_entities(links_st, emb1, emb2, eid_name_dict, args.replace_ratio, hard2easy='Random', cand_num=10, num_thread=16)
    # hard_ent_name_2 = get_hard_entities(links_ts, emb2, emb1, eid_name_dict, args.replace_ratio, hard2easy='Random', cand_num=10, num_thread=16)
    # seq_llm_file_1 = replace_origin_with_enhanced_texts(ent_name_links[:, 0].tolist(), hard_ent_name_1, seq_file_1, llm_file_1)
    # seq_llm_file_2 = replace_origin_with_enhanced_texts(ent_name_links[:, 1].tolist(), hard_ent_name_2, seq_file_2, llm_file_2)
    # emb1 = encode_file_text(model, seq_llm_file_1, encoder_name)
    # emb2 = encode_file_text(model, seq_llm_file_2, encoder_name)
    # emb1, emb2 = reduce_dim_by_SVD(emb1, emb2, reduce_dim=256)
    # print('---- Replace Random ----')
    # eval_alignment_by_sim_mat(emb1, emb2, csls_k=args.CSLS_k)

    # # 消融：选容易的作为困难实体
    # hard_ent_name_1 = get_hard_entities(links_st, emb1, emb2, eid_name_dict, args.replace_ratio, hard2easy='False', cand_num=10, num_thread=16)
    # hard_ent_name_2 = get_hard_entities(links_ts, emb2, emb1, eid_name_dict, args.replace_ratio, hard2easy='False', cand_num=10, num_thread=16)
    # seq_llm_file_1 = replace_origin_with_enhanced_texts(ent_name_links[:, 0].tolist(), hard_ent_name_1, seq_file_1, llm_file_1)
    # seq_llm_file_2 = replace_origin_with_enhanced_texts(ent_name_links[:, 1].tolist(), hard_ent_name_2, seq_file_2, llm_file_2)
    # emb1 = encode_file_text(model, seq_llm_file_1, encoder_name)
    # emb2 = encode_file_text(model, seq_llm_file_2, encoder_name)
    # emb1, emb2 = reduce_dim_by_SVD(emb1, emb2, reduce_dim=256)
    # print('---- Replace Easy ----')
    # eval_alignment_by_sim_mat(emb1, emb2, csls_k=args.CSLS_k)

    # # 消融：困难实体为尾节点
    # hard_ent_name_1 = get_hard_tail_entities(ent_name_links[:, 0].tolist(), seq_file_1, args.replace_ratio)
    # hard_ent_name_2 = get_hard_tail_entities(ent_name_links[:, 1].tolist(), seq_file_2, args.replace_ratio)
    # seq_llm_file_1 = replace_origin_with_enhanced_texts(ent_name_links[:, 0].tolist(), hard_ent_name_1, seq_file_1, llm_file_1)
    # seq_llm_file_2 = replace_origin_with_enhanced_texts(ent_name_links[:, 1].tolist(), hard_ent_name_2, seq_file_2, llm_file_2)
    # emb1 = encode_file_text(model, seq_llm_file_1, encoder_name)
    # emb2 = encode_file_text(model, seq_llm_file_2, encoder_name)
    # emb1, emb2 = reduce_dim_by_SVD(emb1, emb2, reduce_dim=256)
    # print('---- Replace by Resource ----')
    # eval_alignment_by_sim_mat(emb1, emb2, csls_k=args.CSLS_k)