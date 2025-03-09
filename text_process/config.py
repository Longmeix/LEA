import os

text_file_prefix = {'name': 'id_entity',
                    'attr': 'attr_seq',
                    'rel': 'rel_seq',
                    'attr_rel': 'attr_rel_seq',
                    'attr_ngb': 'attr_ngb_seq',
                    }

embs_file_prefix = {'name': f'emb_name',
                    'attr': f'emb_attr_seq',
                    'rel': f'emb_rel_seq',
                    'attr_rel': f'emb_attr_rel_seq',
                    'attr_ngb': f'emb_attr_ngb_seq'}


def init_args(args):
    global text_file, out_embs_prefix, data_dir, save_dir
    llm = args.llm
    input_info = args.input_info.lower()
    data_dir = os.path.join(args.datasets_root, args.dataset)  # if "name" and "no_llm"
    data_dir = os.path.join(data_dir, "output") if (args.input_info != "name" or llm != "no_llm") else data_dir  # if info != "name"
    data_dir = os.path.join(data_dir, f"refine_{llm}") if llm != "no_llm" else data_dir
    save_dir = os.path.join(args.datasets_root, args.dataset, 'output', f'refine_{llm}') # if llm != 'no_llm' else data_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get text file to encode
    text_file = text_file_prefix[input_info]
    if llm != 'no_llm':
        text_file = "name_seq" if input_info == "name" else text_file
        text_file = f'{llm}_sort_' + text_file

    if args.instruct != "normal":
        text_file = f'{args.instruct}_' + text_file

    # get emb file name
    out_embs_prefix = embs_file_prefix[input_info]
    # if llm != 'none':
    out_embs_prefix = f'emb_' + text_file
    

