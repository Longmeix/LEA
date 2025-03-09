import os
from transformers import BartForConditionalGeneration, BartTokenizer
from tools.MultiprocessingTool import MPTool
from tools import FileTools
import concurrent.futures as cf
from tqdm import tqdm
from tools.Announce import Announce

def load_model():
    """加载模型并移动到指定设备"""
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
    return tokenizer, model

def concat_attr_relation_seq(attr_seq_file, rel_seq_file, out_seq_file):
    import pandas as pd
    import csv
    attr_df = pd.read_csv(attr_seq_file, sep='\t', quoting=3, header=None, names=['id', 'ename', 'content']) # quoting=3不去掉内容中的双引号
    rel_df = pd.read_csv(rel_seq_file, sep='\t', quoting=3, header=None, names=['id', 'ename', 'content'])
    merged_df = pd.merge(attr_df, rel_df, on=['id', 'ename'], suffixes=('_attr', '_rel')) # 合并两个dataframe，基于id和ename列
    # 如果attr_df的content与ename相等（没有多余attr），则只rel_df的content，否则合并attr和rel
    merged_df['content'] = merged_df.apply(lambda row: row['content_rel'] if row['content_attr'].rpartition('.')[0] == row['ename'] else row['content_attr']+row['content_rel'], axis=1)
    merged_df = merged_df[['id', 'ename', 'content']].sort_values(by='id')

    merged_df.to_csv(out_seq_file, sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE, quotechar='"', escapechar='\\') # 保留原有的双引号


def PLM_refine_text(text_solver, text_file, model='gpt'):  
    all_text_list = FileTools.load_list(text_file)
    # all_text_list = all_text_list[:2]
    dirname, fname = os.path.split(text_file)
    new_dir = os.path.join(dirname, f'refine_{model}')
    if args.input_info == "name":
        new_dir = os.path.join(dirname, "output", f'refine_{model}') if "output" not in dirname else new_dir
        fname = fname.replace("id_entity", "name_seq")
        
    os.makedirs(new_dir, exist_ok=True) 
    if args.answer_english:
        refine_seq_file = new_dir + f'/english_{model}_{fname}'
        error_seq_file = new_dir + f'/english_error_{fname}'
    else:
        refine_seq_file = new_dir + f'/{model}_{fname}'
        error_seq_file = new_dir + f'/error_{fname}'  # TODO: gpt用英文回答或用原本的语言

    if args.instruct != "normal":
        refine_seq_file = new_dir + f'/{args.instruct}_{model}_{fname}'
        error_seq_file = new_dir + f'/{args.instruct}_error_{fname}'

    have_refined_ids = []
    if os.path.exists(refine_seq_file):
        with open(refine_seq_file, 'r') as fr:
            for line in fr:
                idx = line.strip().split('\t')[0]
                have_refined_ids.append(idx)

    todo_text_list = []  # 文件里已经处理过的id无需再调llm
    for text in all_text_list:
        if text[0] not in have_refined_ids:
            todo_text_list.append(text)

    del all_text_list
    del have_refined_ids

    print(Announce.doing(), 'Refining', refine_seq_file)
    # 多线程，并实时写入文件，结果的序号会乱
    error_seqs = []  # 把敏感、无法用deepseek处理的节点id保存下来，后续用别的模型处理
    with open(refine_seq_file, "a", encoding="utf-8") as file_ref, open(error_seq_file, "a", encoding="utf-8") as file_error:
        with cf.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
            futures = [executor.submit(text_solver, text, model) for text in todo_text_list]
            
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"Using {model} to refine text"):
                try:
                    idx, ent, summary, ori_text = future.result()
                    if summary.endswith('error'):
                        error_seqs.append((idx, ent))
                        file_error.write(f"{idx}\t{ent}\t{ori_text}\n")
                    else:
                        summary = summary.replace('\t', ' ')
                        file_ref.write(f"{idx}\t{ent}\t{summary}\n")
                except Exception as e:
                    raise e
    if len(error_seqs) < 1:  # delete file if empty
        os.remove(error_seq_file)
    print(Announce.done())

    return refine_seq_file, error_seq_file
    

def gpt_entity_explain(id_and_text, model='gpt'):
    from text_process.text2good import gpt_result, deepseek_result, llama_result
    """
    使用GPT检索关于实体的外部信息
    """
    # if args.input_info == "name":
    if len(id_and_text) < 3:
        id_and_text.append("")
    idx, entity, ent_info = id_and_text

    if args.answer_english:
        prompt = f'Based on the following information and your knowledge, explain the entity/code "{entity}" in English, in under 150 words, ' \
                    'covering its definition, common uses, and related concepts. Here are some details: \n'
    else:
        if args.instruct == "simple":
            prompt = f'Explain the entity/term {entity} in English (<150 words). Here are some details: \n'  # prompt简单指令
        elif args.instruct == "short":
            prompt = f'Based on the following information and your knowledge, explain the entity/code "{entity}" in under 50 words, ' \
                        'covering its definition, common uses, and related concepts. Here are some details: \n'  # < 50个词语的prompt
        else:
            prompt = f'Based on the following information and your knowledge, generate a distinctive description for "{entity}" '\
                'in English (<150 words).\n' \
                '**Input Format**:\n' \
                '[ENTITY] (Attributes)\n' \
                'Relations:\n' \
                '-> REL_TYPE -> TAIL ENTITY ([<ATTR>:<VAL>]...)\n' \
                '<- REL_TYPE <- HEAD ENTITY ([<ATTR>:<VAL>]...)\n' \
                'Output Rules:\n' \
                '1. Remove repetitive or useless details.\n' \
                '2. Emphasize distinctive traits compared to similar/neighboring entities.\n' \
                '3. Use neighbor attributes ONLY when boosting distinctiveness.\n' \
    
    if args.input_info == "name":
        prompt = f'Based on the following information and your knowledge, explain the entity/code "{entity}" in English (<150 words), ' \
                        'covering its definition, common uses, and related concepts.'

    
    prompt_text = prompt + bytes(ent_info, "utf-8").decode("unicode_escape")
    if model in ['gpt35', 'gpt4omini', 'gpt4o']:
        os.environ['http_proxy'] = "172.18.166."  # 代理IP
        os.environ['https_proxy'] = os.environ['http_proxy']
        summarized_text = gpt_result(prompt_text, model).replace('\n', '').replace('\r', '').strip() # 确保生成的答案不换行，方便存储
    elif model == 'deepseek':
        summarized_text = deepseek_result(prompt_text).replace('\n', '').replace('\r', '').strip() 
    elif model.startswith("llama"):
        summarized_text = llama_result(prompt_text, model).replace('\n', '').replace('\r', '').strip()
    else:
        raise ValueError(f'No access to LLM {model}')
    return idx, entity, summarized_text, ent_info


def refine_test_candidate_seq(test_link_file, input_info="attr_rel", llm_name="deepseek"):
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

    # 读取测试集实体
    test_links = []
    ancs_1, ancs_2 = [], []
    with open(test_link_file, 'r') as fr:
        for line in fr:
            anc1, anc2 = line.strip('\n').split('\t')
            test_links.append([anc1, anc2])
            ancs_1.append(anc1)
            ancs_2.append(anc2)

    # 对两个网络的测试样本应用LLM
    for net, anc in zip(["1", "2"], [ancs_1, ancs_2]):
        # 原文本句子
        if input_info == "name":
            seq_file = os.path.join(dataset_home, f"id_entity_{net}.txt")
        else:
            seq_file = os.path.join(dataset_home, f"output/{input_info}_seq_{net}.txt")
            # seq_file = os.path.join(dataset_home, f"output/refine_deepseek/deepseek_sort_{input_info}_seq_{net}.txt")
    
        # 读取测试集（候选实体）的原文本句子
        save_test_seq_file = os.path.join(seq_file.replace('.txt', '_test.txt'))
        anc_seq = get_and_save_anc_seq_text(anc, seq_file, save_test_seq_file)  # 保存测试集（候选实体）的原文本句子

        # 用LLM润色或丰富测试集（候选实体）的原文本句子
        refine_seq_file, error_seq_file = PLM_refine_text(gpt_entity_explain, save_test_seq_file, llm_name)
        # 用gpt处理deepseek无法处理的敏感文本
        if os.path.exists(error_seq_file):
            new_llm = "gpt35" if llm_name in ["deepseek"] else llm_name  # deepseek不能处理的敏感数据，用gpt3.5解决；其他因为网络问题的，用原模型重试
            refined_error_file, file_2error = PLM_refine_text(gpt_entity_explain, error_seq_file, model=new_llm)
            assert os.path.exists(file_2error) == False, f"Error file {file_2error} still exists"
            # 合并两个文件
            with open(refine_seq_file, 'a') as f:
                with open(refined_error_file, 'r') as f2:
                    for line in f2:
                        f.write(line)
            if os.path.exists(file_2error): os.remove(file_2error)
        # 读取文件file_ref，并按anc的出现顺序排序后重写file_ref
        get_and_save_anc_seq_text(anc, refine_seq_file, save_match_file=refine_seq_file)

if __name__ == '__main__':
    # # ===== 处理KG的attr变成一段文本 =====
    from preprocess.KBStore import KBStore
    from preprocess.KBConfig import *

    llm_name = args.llm
    input_info = args.input_info
    import torch
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ------ 0. 将KG图转成sequence -----
    dataset1 = Dataset(1)
    dataset2 = Dataset(2)
    print('dataset1:', dataset1)
    print('dataset2:', dataset2)
    fs1 = KBStore(dataset1)
    fs2 = KBStore(dataset2)
    fs1.load_kb()
    fs2.load_kb()
    # data_dir = os.path.dirname(dataset1.attr_seq_out)
    # concat_attr_relation_seq(dataset1.attr_seq_out, dataset1.neighboronly_seq_out, os.path.join(data_dir, f'{input_info}_seq_1.txt'))  # 将attr和relation产生的句子拼起来，更全面的信息
    # concat_attr_relation_seq(dataset2.attr_seq_out, dataset2.neighboronly_seq_out, os.path.join(data_dir, f'{input_info}_seq_2.txt'))
    # # 删除文件
    # os.remove(dataset1.attr_seq_out)
    # os.remove(dataset1.neighboronly_seq_out)
    # os.remove(dataset2.attr_seq_out)
    # os.remove(dataset2.neighboronly_seq_out)
    print('Done')


    test_link_file = os.path.join(dataset_home, '721_5fold/1/test_links')
    refine_test_candidate_seq(test_link_file, input_info=input_info, llm_name=llm_name)