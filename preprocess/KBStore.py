import csv
import re
from collections import defaultdict
from typing import List, Iterator

from tqdm import tqdm

from preprocess.KBConfig import *
from preprocess import Parser
from preprocess.Parser import OEAFileType
from tools import FileTools
from tools.Announce import Announce
from tools.MultiprocessingTool import MPTool
from tools.MyTimer import MyTimer
from tools.text_to_word_sequence import text_to_word_sequence


class KBStore:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.entities = []
        self.literals = []  # attribute values
        self.entity_ids = {}  # entity names to id
        self.classes_ids = {}
        self.literal_ids = {}  # attribute value to id

        self.relations = []
        self.properties = []  # attribute name
        self.relation_ids = {}
        self.property_ids = {}  # attribute name to id

        self.facts = {}  # relation triples
        self.text_facts = {}
        self.literal_facts = {}  # attribute triples
        self.blocks = {}  # attribute triples, value: [entities]
        self.word_level_blocks = {}  # word: [entities]
        self.triple_temp = {} # 记录不在attr entities但在rel entities里出现过的obj，出现两次以上可以算是真正的entity，否则可看作一个attr

        self.relations_inv = []
        self.relation_ids_inv = {}
        self.facts_inv = {}
        self.text_facts_inv = {}

        self.properties_functionality = None
        self.relations_functionality = None

    def load_kb(self) -> None:
        timer = MyTimer()
        # load entities if exists id_ent file
        if os.path.exists(self.dataset.entities_out):
            print("The id_entity file has given")
            self.load_entities()
        # load attr
        if args.attribute:
            self.load_path(self.dataset.attr, self.load, OEAFileType.attr)
        if args.relation:
            self.load_path(self.dataset.rel, self.load, OEAFileType.rel)
            self.relations_functionality = KBStore.calculate_func(self.relations, self.relation_ids, self.facts,
                                                                  self.entity_ids)
            # 增加对facts按relation排序
            for ent_id, facts in self.facts.items():
                facts: list
                r_facts = []
                for r_id, obj_id in facts:
                    r_facts.append((r_id, obj_id, self.relations_functionality[r_id]))
                r_facts.sort(key=lambda x: (-x[2], x[0], x[1]), reverse=False)
                self.facts[ent_id] = r_facts
            # self.facts = sorted(self.facts.items(), key=lambda item: item[0])

        if args.attribute:
            self.properties_functionality = KBStore.calculate_func(self.properties, self.property_ids, self.literal_facts,
                                                               self.entity_ids)
            # 增加对literal_facts按properties_functionality重要性排序
            # aa = {}
            for ent_id, each_attr_facts in self.literal_facts.items():
                l_facts = []
                facts: list
                for i, a_fact in enumerate(each_attr_facts):
                    a_property_id, a_val = a_fact
                    l_facts.append((a_property_id, a_val, self.properties_functionality[a_property_id]))
                    # importance = (1 - self.properties_functionality[a_property_id]) +
                l_facts.sort(key=lambda x: (-x[2], x[0], x[1]), reverse=False)  # 按attr重要性从大到小排序, functionality越小越重要
                self.literal_facts[ent_id] = l_facts

        timer.stop()

        print(Announce.printMessage(), 'Finished loading in', timer.total_time())
        if not os.path.exists(self.dataset.entities_out):
            self.save_base_info()
        self.save_datas()

    def save_base_info(self):
        print(Announce.doing(), 'Save Base Info')
        FileTools.save_dict_reverse(self.entity_ids, self.dataset.entities_out)
        # FileTools.save_dict_reverse(self.literal_ids, self.dataset.literals_out)
        print(Announce.done(), 'Finished Saving Base Info')
    
    def get_property_table_line(self, line):
        e, ei = line
        e: str
        if type(ei) != int:
            print(line)
        # ename = e.split('/')[-1]
        ename = e
        dic = {'ent_id': ei, 'ent_name': ename}
        facts = self.literal_facts.get(ei)
        if facts is not None:
            fact_aggregation = defaultdict(list)
            for fact in facts:
                # 过滤函数性低的
                if functionality_control and self.properties_functionality[fact[0]] <= functionality_threshold:
                    continue
                fact_aggregation[fact[0]].append(self.literals[fact[1]])

            for pid, objs in fact_aggregation.items():
                pred = self.properties[pid]
                objs = [objs] if isinstance(objs, str) else objs
                new_objs = []
                for a_val in objs:
                    if len(a_val) > 20 and len(facts) > 10:  #有10个属性以上，可以丢弃每个属性中太长的值
                        a_val = ' '.join(a_val.split(' ')[:5])
                        continue
                    new_objs.append(a_val)
                dic[pred] = list(new_objs)
                # # obj = ', '.join(objs)
                # dic[pred] = objs

        return dic

    def save_property_table(self):  # 列为每个attribute name, 行为每个实体的表格
        # table_path = self.dataset.table_out
        # print(Announce.doing(), 'Save', table_path)
        
        header = ['ent_id', 'ent_name']
        if not functionality_control:
            header.extend(self.property_ids.keys())
        else:
            for p, pid in self.property_ids.items():
                if self.properties_functionality[pid] > functionality_threshold:
                    header.append(p)
        dicts = MPTool.packed_solver(self.get_property_table_line).send_packs(
            self.entity_ids.items()).receive_results()
        # dicts = filter(lambda dic: dic is not None, dicts)
        # dicts = list(dicts)

        # print(Announce.done())
        return dicts, header

    def get_relation_table_line(self, line, relations):
        eid, rel_facts = line
        if type(eid) != int:
            print(line)
        dic = {'ent_id': eid, 'ent_name': self.entities[eid]}
        if rel_facts is not None:
            fact_aggregation = defaultdict(list)
            for fact in rel_facts:
                fact_aggregation[fact[0]].append(self.entities[fact[1]])

            for rid, objs in fact_aggregation.items():
                # rel = self.relations[rid]
                rel = relations[rid]
                objs = [objs] if isinstance(objs, str) else objs
                dic[rel] = objs
                # obj = ', '.join(objs)
                # dic[rel] = obj

        return dic
    
    def save_relation_table(self, triples_dict, relations):  # 列为每个relation name, 行为每个实体的表格
        from functools import partial
        get_relation_with_params = partial(self.get_relation_table_line, relations=relations)
        dicts = MPTool.packed_solver(get_relation_with_params).send_packs(
            triples_dict.items()).receive_results()
        dicts = filter(lambda dic: dic is not None, dicts)
        dicts = list(dicts)
        nest_dict = {}
        for dic in dicts:
            nest_dict[dic['ent_id']] = dic

        return nest_dict

    def save_triple_seq_form(self):  # here save sequence for entity
        def get_neighboronly_seq(ent_id: dict):
            def count_words(s):
                # 使用正则表达式分割字符串，\W+ 表示非字母数字的字符序列
                words = re.findall(r'\b\w+\b', s)
                return len(words)
            
            ename, eid = ent_id
            # head实体通过关系指向的邻居
            seq = ""

            ent_neighbors_dict = self.rel_neighbors_dict.get(eid, None)
            if ent_neighbors_dict:  # 如果head有指向的邻居实体
                assert ename == ent_neighbors_dict['ent_name']
                rel_seq = 'its '
                top_neighbors_dict = ent_neighbors_dict
                if len(ent_neighbors_dict) > 10:
                    top_neighbors_dict = dict(list(ent_neighbors_dict.items())[:10]) # 选择重要的邻居

                for rel, neighbors in top_neighbors_dict.items(): 
                    if rel not in ['ent_id', 'ent_name']:
                        # neighbors = str(neighbors).split(', ')
                        rel_seq += f"{rel} is {' and '.join(neighbors)}, "  # 列举
                        # if len(neighbors) > 3:  # 同一个关系连接的tail节点太多，则只保留这个relation和部分实体
                        #     rel_out_part_ents[rel] = 'and '.join(neighbors[:3])
                        #     continue
                        # else:
                        #     rel_seq += f"{rel} is {'and '.join(neighbors)}, "  # 列举
                if len(rel_seq) > 4:
                    rel_seq = rel_seq[:-2] + '. '    # 整理格式 
                    seq = f'"{ename}" has some relations with other terms/entities: ' + rel_seq
                # if len(only_rel_name) > 0 and :  # 优先加概括的关系信息还是 下面精确的反向信息？
                #     rel_seq += f'Besides, "{ename}" has relations ' + ', '.join(only_rel_name)  
            
            if count_words(seq) < 50:  # 正向邻居的信息过少，则补充反向邻居信息，即指向head的信息
                neighbors_to_head_dict = self.rel_inv_neighbors_dict.get(eid, None)
                if neighbors_to_head_dict:
                    assert ename == neighbors_to_head_dict['ent_name']
                    rel_seq = ''
                    # only_rel_name = []
                    top_neighbors_dict = neighbors_to_head_dict
                    if len(neighbors_to_head_dict) > 10:
                        top_neighbors_dict = dict(list(neighbors_to_head_dict.items())[:10]) # 选择重要的反向邻居
                    for rel, neighbors in top_neighbors_dict.items(): 
                        if rel not in ['ent_id', 'ent_name']:
                            rel = rel.split('-inv')[0]
                            # neighbors = str(neighbors).split(', ')
                            pred = 'are' if len(neighbors) > 1 else 'is'
                            rel_seq += f"{' and '.join(neighbors[:3])}\'s {rel} {pred} \"{ename}\", "  # 列举
                            # if len(neighbors) > 3:  # 其他实体指向head的太多，则只保留relation不保留值
                            #     rel_in_part_ents[rel] = 'and '.join(neighbors[:3])
                            #     continue
                            # else:
                            #     pred = 'are' if len(neighbors) > 1 else 'is'
                            #     rel_seq += f"{'and '.join(neighbors)}\'s {rel} {pred} \"{ename}\", "  # 列举
                    if len(rel_seq) > 4:
                        rel_seq = rel_seq[:-2] + '. '    # 整理格式
                        seq = f'"{ename}" has some relations with other terms/entities: ' + rel_seq if len(seq) < 1 else seq + rel_seq
            
            # assert len(seq) > 0
            seq = ename+'.' if len(seq) < 1 else seq
            return eid, ename, seq

        seq_path = self.dataset.neighboronly_seq_out
        print(Announce.doing(), 'Save', seq_path)
        seqs = MPTool.packed_solver(get_neighboronly_seq).send_packs(self.entity_ids.items()).receive_results()
        FileTools.save_list(seqs, seq_path)

        print(Announce.done())

    def save_attr_rel_seq_form(self):  # here save sequence for entity
        def get_neighboronly_seq(ent_id: dict):
            def sort_neighbors_by_num_of_attribute(rel_neighbors, attr_dicts):
                sorted_rel_neighbors = []
                for idx, (rel, neighbors) in enumerate(rel_neighbors.items()):
                    has_attibute = 0
                    neighbor_list = str(neighbors).split(', ')
                    for nei_name in neighbor_list:
                        nid = self.entity_ids.get(nei_name)
                        nei_attrs = attr_dicts.get(nid, {})
                        if len(nei_attrs) > 0:
                            has_attibute = 1
                            break
                    sorted_rel_neighbors.append((rel, neighbors, has_attibute, idx))
                sorted_rel_neighbors.sort(key=lambda x: (-x[2], x[3]))
                return {rel: neighbors for rel, neighbors, _, _ in sorted_rel_neighbors}
            
            ename, eid = ent_id
            seq = ""
            # 实体的attribute dict, 键为实体id
            head_attr = dict(list(attr_dicts.get(eid, {}).items())[:10]) 

            num_rel = 4
            if len(head_attr) > 5:
                num_rel = 3  # 属性多的话，可只取前3个关系，使输入LLM的文本不过多

            # 获得该实体的邻居
            ent_neighbors_dict = {**self.rel_neighbors_dict.get(eid, {}), **self.rel_inv_neighbors_dict.get(eid, {})}
            if ent_neighbors_dict:  # ent作为head
                assert ename == ent_neighbors_dict['ent_name']
                # 优先选择具有属性的邻居
                ent_neighbors_dict = dict(list(ent_neighbors_dict.items())[2:]) # 去掉前两个['ent_id', 'ent_name']
                ent_neighbors_dict = sort_neighbors_by_num_of_attribute(ent_neighbors_dict, attr_dicts)  # 排序邻居，优先选有属性的邻居
                top_neighbors_dict = dict(list(ent_neighbors_dict.items())[:num_rel])  # 选择前n个重要的关系
                seq += f'Entity [{ename}]' + (f' has attributes: ({head_attr})\n' if len(head_attr) > 0 else '\n')
                seq += 'Relations: \n'
                for rel, neighbors in top_neighbors_dict.items(): 
                    is_head2tail = False if '-inv' in rel else True  # 反向关系
                    seq += f"-> {rel} -> " if is_head2tail else f"<- {rel.split('-inv')[0]} <- "
                    # neighbors = str(neighbors).split(', ')[:2]  # 一种关系只取前2个邻居
                    neighbors = neighbors[:2] # 一种关系只取前2个邻居
                    # 加入邻居的相关属性
                    rel_ngbs = []
                    for nei_name in neighbors:
                        nid = self.entity_ids.get(nei_name)
                        nei_attrs = attr_dicts.get(nid, {})
                        if len(nei_attrs) > 3:  # 取部分属性
                            nei_attrs = dict(list(nei_attrs.items())[:3])
                        rel_ngbs.append(f"{nei_name}" + (f" ({nei_attrs})" if len(nei_attrs) > 0 else ''))  # “邻居 (属性)”
                    rel_ngbs = ', '.join(rel_ngbs)
                    seq += rel_ngbs + '\n'
            else:
                raise Exception(f"Error: {ename} is a isolated node.")
            
            # assert len(seq) > 0
            seq = ename+'.' if len(seq) < 1 else seq
            # seq = repr(seq)  # 使用 repr() 转义换行符，将一个实体的信息显示在一行
            seq = seq.replace('\n', '\\n')
            return eid, ename, seq

        seq_path = self.dataset.attr_rel_seq_out
        print(Announce.doing(), 'Save', seq_path)

        # 筛选有用属性，避免太多消耗tokens
        keys_to_remove = ['ent_id', 'ent_name']
        attr_dicts = {}
        for key, sub_dict in enumerate(self.attr_dicts):
            ent_attrs = list(sub_dict.items())[:12]  # 最多存10个属性
            new_sub_dict = {k: ', '.join(v[:2]) for k, v in ent_attrs if k not in keys_to_remove}  # 同一个属性只取前两个值，存在'购买力平价GDP': '977529225584, 1545932373238, 2260384821849, 2329213712304'意义小占长度
            # new_sub_dict = {k: v for k, v in sub_dict.items() if k not in keys_to_remove} 
            attr_dicts[key] = new_sub_dict
        seqs = MPTool.packed_solver(get_neighboronly_seq).send_packs(self.entity_ids.items()).receive_results()
        FileTools.save_list(seqs, seq_path)

        print(Announce.done())

    def save_attr_seq_form(self, dicts: Iterator, header: List):
        def get_useful_value(vals):
            if len(vals) > 11 and bool(re.search(r'(?:\d.*?){11,}', vals)): # 达到11个数字
                return None
            
            # useful_vals = []
            # for val in vals.split(', '):  # 提取每个属性值的有效值
            #     toks = text_to_word_sequence(val)
            #     for tok in toks:
            #         if len(tok) < 5: # 长度小于5的attr value都保留
            #             continue
            #         if bool(re.search(r'\d', tok)):  # 丢弃属性值太长且含有数字的属性
            #             toks = ''
            #             break
            #     if toks == '':
            #         continue
            #     else:
            #         useful_vals.append(val)
            # useful_vals = ', '.join(useful_vals) if len(useful_vals) > 0 else None
            # return useful_vals
            return vals
                
        def get_attr_seq(dic: dict):
            eid, ename = dic['ent_id'], dic['ent_name']
            h = header.copy()[1:]
            # values = [str(dic[key]) for key in h if key in dic]  # 所有attribute value的序列
            seq = f""
            attr = ''
            only_attr_name = []
            top_attr_dict = dic
            if len(top_attr_dict) > 10:
                top_attr_dict = dict(list(top_attr_dict.items())[:10]) # 选择重要的属性
            for key, vals in top_attr_dict.items():
                if key not in ['ent_id', 'ent_name']:
                    if len(vals) > 3:  # 同一个属性连接的实体太多，则只保留部分
                        vals = get_useful_value(', '.join(vals))
                    else:
                        vals = ', '.join(vals)
                    if vals != None:
                        attr += f"{key}: {vals}, "
                    else:
                        only_attr_name.append(key)
                        continue
            if len(attr) > 0:
                seq = f'"{ename}" has the following attributes: ' + attr[:-2] + '. ' # [:-2]去掉最后的逗号和空格
            if len(only_attr_name) > 0:  # 部分属性值太多或意义不大，只打印属性名
                other_attr = ', '.join(only_attr_name)
                if len(seq) > 0:
                    seq += f'Besides, "{ename}" has other attributes such as {other_attr}. '
                else:
                    seq += f'"{ename}" has the following attributes: {other_attr}. '
            seq = ename+'. ' if len(seq) < 1 else seq  # 没有属性则输出实体名称
            return eid, ename, seq
        
        seq_path = self.dataset.attr_seq_out
        print(Announce.doing(), 'Save', seq_path)
        seqs = MPTool.packed_solver(get_attr_seq).send_packs(dicts).receive_results()
        # seqs = [get_seq(dic) for dic in dicts]
        FileTools.save_list(seqs, seq_path)

    def save_datas(self):
        print(Announce.doing(), 'Save data2')

        # if args.relation:  # 存储 关系id, relation name, 关系重要性
        #     print(Announce.printMessage(), 'Save', self.dataset.relations_out)
        #     with open(self.dataset.relations_out, 'w', encoding='utf-8') as wfile:
        #         for r, id in self.relation_ids.items():
        #             print(id, r, self.relations_functionality[id], sep='\t', file=wfile)

        # 保存property csv
        if args.attribute:
            self.attr_dicts, header = self.save_property_table()
            self.save_attr_seq_form(self.attr_dicts, header)
        if args.relation:
            self.rel_neighbors_dict = self.save_relation_table(self.facts, self.relations)  # 把head相同关系的tail实体放在一起 {实体e：{关系i：[相连的邻居们]}}
            self.rel_inv_neighbors_dict = self.save_relation_table(self.facts_inv, self.relations_inv)
            self.save_triple_seq_form()
        # if args.relation and args.attribute:
        #     self.save_attr_rel_seq_form()
        print(Announce.done(), 'Finished')

    def load_kb_from_saved(self):
        self.load_entities()
        # self.load_literals()
        self.load_relations()
        self.load_properties()
        self.load_facts()
        pass

    def load_entities(self):
        print(Announce.doing(), 'Load entities', self.dataset.entities_out)
        # self.entity_ids = FileTools.load_dict_reverse(self.dataset.entities_out)
        entity_list = FileTools.load_list(self.dataset.entities_out)
        self.entity_ids = {ent: int(s_eid) for s_eid, ent in entity_list}
        self.entities = [ent for s_eid, ent in entity_list]
        print(Announce.done())
        pass

    def load_relations(self):
        relation_list = FileTools.load_list_p(self.dataset.relations_out)
        self.relation_ids = {rel: rid for rid, rel, func in relation_list}
        self.relations = [rel for rid, rel, func in relation_list]

    def load_properties(self):
        property_list = FileTools.load_list(self.dataset.properties_out)
        self.property_ids = {prop: s_pid for s_pid, prop, s_func in property_list}

    def load_facts(self):
        fact_list = FileTools.load_list_p(self.dataset.facts_out)
        self.facts = {eid: elist for eid, elist in fact_list}

    @staticmethod
    def save_blocks(fs1, fs2):
        if args.blocking:
            pass
        pass

    @staticmethod
    def load_path(path, load_func, file_type: OEAFileType) -> None:
        print(Announce.doing(), 'Start loading', path)
        if os.path.isdir(path):
            for file in sorted(os.listdir(path)):
                if os.path.isdir(file):
                    continue
                file = os.path.join(path, file)
                # load_func(file, type)
                KBStore.load_path(file, load_func, file_type)
        else:
            load_func(path, file_type)
        print(Announce.done(), 'Finish loading', path)

    def load(self, file: str, file_type: OEAFileType) -> None:
        tuples = Parser.for_file(file, file_type)
        with tqdm(desc='add tuples', file=sys.stdout, ncols=80) as tqdm_add:
            tqdm_add.total = len(tuples)
            for args in tuples:
                self.add_tuple(*args, file_type)
                tqdm_add.update()
        pass

    def add_tuple(self, sbj: str, pred: str, obj: str, file_type: OEAFileType) -> None:
        assert sbj is not None and obj is not None and pred is not None, 'sbj, obj, pred None'
        if file_type == OEAFileType.attr:
            if obj.startswith('"'):
                obj = obj[1:-1]
            toks = text_to_word_sequence(obj)
            for tok in toks:
                if len(tok) < 5: # 长度小于5的attr value都保留
                    continue
                # if bool(re.search(r'\d', tok)):  # 丢弃数字属性值太长的属性  
                #     return
            sbj_id = self.get_or_add_item(sbj, self.entities, self.entity_ids)  # entity 以attr的实体数量为准
            obj_id = self.get_or_add_item(obj, self.literals, self.literal_ids)  # attribute value
            pred_id = self.get_or_add_item(pred, self.properties, self.property_ids)  # attribute name
            self.add_fact(sbj_id, pred_id, obj_id, self.literal_facts) # self.literal_facts=[[一个head节点出发的(rel,tail)], [(r1, t1), (r2,t2)], ]
            self.add_to_blocks(sbj_id, obj_id)  # self.block=[0: {指向尾节点的所有head id}]
            words = text_to_word_sequence(obj)
            self.add_word_level_blocks(sbj_id, words)
        elif file_type == OEAFileType.rel:
            # for zh_vi等低资源语言数据集
            sbj_id = self.get_or_add_item(sbj, self.entities, self.entity_ids)
            # if add_obj_ent:  # TODO: 当为DBP数据集时，ents列表不加入obj实体，因为对齐节点时没用到
            obj_id = self.get_or_add_item(obj, self.entities, self.entity_ids)  
            # else:
            #     obj_id = self.get_or_add_item(obj, self.literals, self.literal_ids)  # attribute value
            #     pred_id = self.get_or_add_item(pred, self.properties, self.property_ids)  # attribute name
            if obj in self.entity_ids:  # 
                obj_id = self.entity_ids.get(obj)
                pred_id = self.get_or_add_item(pred, self.relations, self.relation_ids)
                # pred2_id = self.get_or_add_item(pred + '-inv', self.relations, self.relation_ids)  
                pred2_id = self.get_or_add_item(pred + '-inv', self.relations_inv, self.relation_ids_inv)  
                self.add_fact(sbj_id, pred_id, obj_id, self.facts)  # relation
                # self.add_fact(obj_id, pred2_id, sbj_id, self.facts) 
                self.add_fact(obj_id, pred2_id, sbj_id, self.facts_inv) 
            # self.add_fact(sbj, pred, obj, self.text_facts)  # 
            # # self.add_fact(obj, pred + '-inv', sbj, self.text_facts) 
            # self.add_fact(obj, pred + '-inv', sbj, self.text_facts_inv) 

            # # for 
            # sbj_id = self.get_or_add_item(sbj, self.entities, self.entity_ids)
            # if sbj in self.triple_temp:  # triple_temp里保存只在obj出现过一次的实体
            #     del self.triple_temp[sbj]
            # if obj not in self.entity_ids:
            #     if obj in self.triple_temp:  # 如果obj出现过两次，可作为真正的实体加入网络
            #         obj_id = self.get_or_add_item(obj, self.entities, self.entity_ids)
            #         # 并且把第一次出现的关系三元组添加进来
            #         sbj1, pred1, obj1 = self.triple_temp[obj]
            #         self.add_facts_and_text_facts(sbj1, pred1, obj1)
            #         del self.triple_temp[obj]
            #     else: # obj第一次出现，先记录，再确认是attr的value还是网络里的entity
            #         self.triple_temp[obj] = (sbj, pred, obj)
            # if obj in self.entity_ids: # 加入本次关系三元组
            #     self.add_facts_and_text_facts(sbj, pred, obj)

    def add_item(self, name: str, names: list, ids: dict) -> int:
        iid = len(names)
        names.append(name)
        ids[name] = iid
        return iid

    def get_or_add_item(self, name: str, names: list, ids: dict) -> int:
        if name in ids:
            return ids.get(name)
        else:
            return self.add_item(name, names, ids)

    def add_fact(self, sbj_id, pred_id, obj_id, facts_list: dict) -> None:
        if sbj_id in facts_list:
            facts: list = facts_list.get(sbj_id)
            facts.append((pred_id, obj_id))
        else:
            facts_list[sbj_id] = [(pred_id, obj_id)]

    def add_to_blocks(self, sbj_id, obj_id) -> None:
        if obj_id in self.blocks:
            block: set = self.blocks.get(obj_id)
            block.add(sbj_id)
        else:
            self.blocks[obj_id] = {sbj_id}

    def add_word_level_blocks(self, entity_id, words):
        for word in words:
            if word in self.word_level_blocks:
                block: set = self.word_level_blocks.get(word)
                block.add(entity_id)
            else:
                self.word_level_blocks[word] = {entity_id}
        pass

    @staticmethod
    def calculate_func(r_names: list, r_ids: dict, facts_list: dict, sbj_ids: dict) -> list:
        num_occurrences = [0] * len(r_names)
        func = [0.] * len(r_names)
        num_subjects_per_relation = [0] * len(r_names)
        last_subject = [-1] * len(r_names)

        for sbj_id in sbj_ids.values():
            facts = facts_list.get(sbj_id)
            if facts is None:
                continue
            for fact in facts:
                num_occurrences[fact[0]] += 1
                if last_subject[fact[0]] != sbj_id:
                    last_subject[fact[0]] = sbj_id
                    num_subjects_per_relation[fact[0]] += 1

        for r_name, rid in r_ids.items():
            func[rid] = num_subjects_per_relation[rid] / num_occurrences[rid]
            # print(Announce.printMessage(), rid, r_name, func[rid], sep='\t')
        return func