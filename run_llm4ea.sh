# --- change them ---
datasets_root="./data/standard"
dataset="doremus_en" # choices=['zh_en', 'zh_vi', 'icews_wiki', 'doremus_en']
llm="deepseek"  # choices=['no_llm', 'llama_8b', 'llama_70b', 'deepseek', 'gpt4omini']
input_info="attr_ngb"  # choices=['attr_ngb', 'attr', 'rel', 'name' ]
encoder="JINAv3"  # ["BERT", "sBERT", "BGEm3", "JINAv3"]
instruct="normal"
# instruct="simple"

max_threads=8
# answer_english=false  # 暂时不改

# --- hyperparameters
single_thread_list=("gpt4omini", "gpt4o", "gpt35")
if [[ "${single_thread_list[@]}" =~ "${llm}" ]]; then
    max_threads=1
fi

paras="--datasets_root $datasets_root --dataset $dataset --llm $llm --input_info $input_info --instruct $instruct"
if [[ $answer_english == "true" ]]; then
    paras="$paras --answer_english"
fi

# --- running ---
start_time=$(date +%s)

echo "### 1. Enrich KG-aware sequences by llm, threads=$max_threads "
python -u lea_main.py $paras --max_threads $max_threads

echo "### 2. Encoding sequences into embeddings and alignment by $encoder"
python -u EntMatcher/encode_align.py $paras --encoder $encoder


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total time: $elapsed_time s"
