import os



def PLM_refine_text(raw_texts):  
    def summarize_text(model, tokenizer, text):
        """
        使用BART模型对输入文本生成摘要
        """
        inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    from transformers import BartForConditionalGeneration, BartTokenizer
    # 加载BART的预训练模型和分词器
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # # 对每一行文本进行摘要生成  单条串行执行
    # for i, text in enumerate(raw_texts):
    #     print(f"Original Text {i+1}: {text}\n")
    #     summarized_text = summarize_text(text)
    #     print(f"Summarized Text {i+1}: {summarized_text}\n")

    # 并行执行
    import concurrent.futures as cf
    from functools import partial
    from tqdm import tqdm
    summarized_text = []
    # partial_summarize = partial(summarize_text, model, tokenizer)
    # 使用线程池并行处理
    with cf.ThreadPoolExecutor() as executor:
        # 使用 submit 提交所有任务
        futures = [executor.submit(summarize_text, model, tokenizer, text) for text in raw_texts]
        # 使用tqdm可视化进度条
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Refine Text"):
            summarized_text.append(future.result())

    for i, text in enumerate(raw_texts):
        print(f"Original Text {i+1}: {text}\n")
        print(f"Summarized Text {i+1}: {summarized_text[i]}\n")



# def deepseek_result(prompt):

#     '''
#     通过DeepSeek官方调用API
#     '''

#     from text_process.gpt_request import deepseek_request
#     model = "deepseek-chat"
#     system_msg = "You are an intelligent assistant!"  # 系统设置，每次对话都会自动加上这句话
#     # prompt = ""  # 用户输入的话

#     # 构建输入
#     if system_msg is None:
#         messages = [
#             {"role": "user", "content": prompt},
#         ]
#     else:
#         messages = [
#             {"role": "system", "content": system_msg},
#             {"role": "user", "content": prompt},
#         ]

#     temperature = 1.0
#     stop = None
#     top_p = 1
#     max_tokens = 1024
#     n = 1  # 对于一个query，模型可以同时输出n个相互独立的回答

#     res = deepseek_request(model, messages, temperature, stop=stop, top_p=top_p, max_tokens=max_tokens, n=n)

#     return res



def deepseek_result(prompt) -> str:
    '''
    通过硅基流动平台调用deepseek模型，以字符串格式返回
    '''

    import random
    import requests
    from text_process.gpt_request import siliconflow_request_chat

    api_keys = [
        "sk-nggivsgdckrdbjicqsotfqveirfitkpdeeihuuogvhkafxhb", # 骏骁API  119
        "sk-etvtdkwmwbbehljqiiwohygspdbfmamnalawuoobkmocehes", # 自通·1  83
        "sk-qbdzsexifucqnviyzxuolsmrsarbsemlwybvrsnoptnfgbtz" # 自通·2  104
    ]
    api_keys = random.choice(api_keys)
    model = "deepseek-ai/DeepSeek-V2.5"

    system_msg = "You are an intelligent assistant!"  # 系统设置，每次对话都会自动加上这句话

    # 构建输入
    if system_msg is None:
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

    temperature = 1.0 # 该部分超参数自deepseek配置复制而来
    stop = None
    top_p = 1
    max_tokens = 1024
    n = 1  # 对于一个query，模型可以同时输出n个相互独立的回答

    try:
        chat_completion = siliconflow_request_chat(api_keys, model=model, messages=messages, temperature=temperature, stop=stop, top_p=top_p, max_tokens=max_tokens, n=n)
        result = chat_completion["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        print("请求超时！")
        result = "error"
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP错误: {http_err}")
        result = "error"
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        result = "error"

    return result



def gpt_result(prompt, model_name):
    import time
    from text_process.gpt_request import gpt_request
    models = {"gpt35": "gpt-3.5-turbo", 
              "gpt4o": "gpt-4o",
              "gpt4omini": "gpt-4o-mini"}
    model = models[model_name]
    # model = "deepseek-chat"
    system_msg = "You are an intelligent assistant!"  # 系统设置，每次对话都会自动加上这句话
    # prompt = ""  # 用户输入的话

    # 构建输入
    if system_msg is None:
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

    temperature = 1.0
    stop = None
    top_p = 1
    max_tokens = 1024
    n = 1  # 对于一个query，模型可以同时输出n个相互独立的回答

    res = gpt_request(model, messages, temperature, stop=stop, top_p=top_p, max_tokens=max_tokens, n=n)
    # # 解析模型输出
    # if n == 1:
    #     res = response["choices"][0]["message"]["content"].strip()
    # else:
    #     res = [r["message"]["content"].strip() for r in response["choices"]]

    # retries = 0
    # try:
    #     response = gpt_request(model, messages, temperature, stop=stop, top_p=top_p, max_tokens=max_tokens, n=n)
        
    #     # 检查response中是否存在'choices'键，并且该键的值是一个非空列表
    #     if "choices" in response and response["choices"]:
    #         if n == 1:
    #             res = response["choices"][0]["message"]["content"].strip()
    #             return res
    #         else:
    #             res = [r["message"]["content"].strip() for r in response["choices"]]
    #             return res
    #     else:
    #         # 如果'choices'不存在或为空，则记录错误并重试
    #         # print("======", str(retries) + " input: " + prompt)
    #         retries += 1
    #         if retries < max_retries:
    #             time.sleep(2)  # 等待一段时间再重试
    #         else:
    #             return response
    #             # raise ValueError("No choices found in response after multiple retries.")
    # except Exception as e:
    #     # 处理其他可能的异常
    #     print(f"An error occurred: {e}")
    return res


def llama_result(prompt, model_name) -> str:
    '''
    通过硅基流动平台调用llama模型，以字符串格式返回
    '''

    import random
    import requests
    from text_process.gpt_request import siliconflow_request_chat

    api_keys = [
        "sk-nggivsgdckrdbjicqsotfqveirfitkpdeeihuuogvhkafxhb", # 骏骁API  119
        "sk-etvtdkwmwbbehljqiiwohygspdbfmamnalawuoobkmocehes", # 自通·1  83
        "sk-qbdzsexifucqnviyzxuolsmrsarbsemlwybvrsnoptnfgbtz" # 自通·2  104
    ]

    model_name_map = {"llama_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct", "llama_70b": "meta-llama/Meta-Llama-3.1-70B-Instruct"} # 模型名称映射表
    model = model_name_map.get(model_name, model_name) # 通过查表找到平台对应的模型名称，找不到则直接使用输入的内容

    system_msg = "You are an intelligent assistant!"  # 系统设置，每次对话都会自动加上这句话

    # 构建输入
    if system_msg is None:
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

    temperature = 1.0 # 该部分超参数自deepseek配置复制而来
    stop = None
    top_p = 1
    max_tokens = 1024
    n = 1  # 对于一个query，模型可以同时输出n个相互独立的回答

    try:
        chat_completion = siliconflow_request_chat(random.choice(api_keys), model=model, messages=messages, temperature=temperature, stop=stop, top_p=top_p, max_tokens=max_tokens, n=n)
        result = chat_completion["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        print("请求超时！")
        result = "error"
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP错误: {http_err}")
        result = "error"
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        result = "error"

    return result


def gpt_summarize_text(raw_texts):
    """
    使用GPT对输入文本生成摘要
    """
    # prompt = "Summarize the following text in a more coherent and concise way:\n"
    prompt = "Rewrite the following unstructured and verbose text into a coherent, high-quality summary. \
                Retain the key points and, based on your knowledge, enhance the content by adding relevant details if necessary. \
                Make sure the output is coherent and concise. \
                Here is the original text: \n"
    # 对每一行文本进行摘要生成
    for i, text in enumerate(raw_texts):
        print(f"Original Text {i+1}: {text}\n")
        prompt_text = prompt + text
        summarized_text = gpt_result(prompt_text)
        print(f"Summarized Text {i+1}: {summarized_text}\n")



if __name__ == '__main__':
    # # 原始文本列表，每行代表一个要总结的文本
    # raw_texts = [
    #     "Nb3X8 and VcF in factor-2 and HADH and INCL and Fba and CTNT causing traumatic_occlusion and hypoplastic_right_heart_syndrome and traumatic_occlusion and SIV-infected and poison_ivy and SIV-infected and Tx and SIV-infected and hospital_infections and Juvenile_dermatomyositis and cerebral_death and Juvenile_dermatomyositis and mediastinal_lymphadenopathy and cystic_malformations and Trachea",
    #     "DPC in Acute_megakaryoblastic_leukemia and Trachea and rd and MSDs and pancreatic_dysfunction and MSDs and SPNs and visual_illusions and SPNs and prepatellar_bursitis and cerebrovascular_stroke and overuse_syndrome and cerebrovascular_stroke and encephalon and PTGS and RSDB and SIV-infected and tuberculosis_(TB)_infection and SIV-infected and prolonged_QTc_duration and Androgenetic and seminomatous and Androgenetic and ESLD and Androgenetic and bacterial_and_fungal_cultures",
    #     "hTG in streptococcal_infection and activity-stress and streptococcal_infection and adult_glioblastoma and streptococcal_infection and CRIF and streptococcal_infection and activity-stress and streptococcal_infection and PS+SAP and neurofibrillary_tangles and frontal_lobe_syndrome and colonic_segments and congestive_cardiac_failure and LDLr(-/-)",
    #     "CCR7 causing cryptorchism"
    #     ]

    # 查余额
    # import requests
    # url = "https://api.siliconflow.cn/v1/user/info"
    # # api_key = "sk-nggivsgdckrdbjicqsotfqveirfitkpdeeihuuogvhkafxhb" # 骁API  119  112
    # # api_key = "sk-etvtdkwmwbbehljqiiwohygspdbfmamnalawuoobkmocehes" # 通·1  83  76
    # api_key = "sk-qbdzsexifucqnviyzxuolsmrsarbsemlwybvrsnoptnfgbtz" # 通·2  104  97.5
    # headers = {
    #   "Accept": "application/json",
    #   "Authorization": f"Bearer {api_key}"
    # }
    # response = requests.request("GET", url, headers=headers)
    # print(response.text)
    
    text = f'Based on the following information and your knowledge, explain what the entity/code "保佑这高尚的土地" is in English, ' \
            'including its definition, the field it belongs to, common uses, or related concepts. ' \
            'Make sure the output is coherent and concise, in under 200 words. Here are some details: \n 波札那\'s 颂歌 is "保佑这高尚的土地". '
    
    print(text)
    res = deepseek_result(text)
    print(res)
