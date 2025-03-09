import requests
import time
import typing

def gpt_request(model, final_input, temperature, top_p=1, max_tokens=1024, frequency_penalty=0, presence_penalty=0,
                n=1, stop=None):
    url = "https://api.pumpkinaigc.online/v1/chat/completions"
    api_keys = [
        "sk-8zrnDuUQ",
    ]

    data = {
        'model': model,
        'messages': final_input,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty,
        'n': n,
        'stop': stop
    }

    num_key = len(api_keys)
    key_id = 0
    # for i in range(num_key):
    headers = {
        # "Authorization": f"Bearer {api_keys[key_id % num_key]}",
        "Authorization": f"Bearer {api_keys[0]}",
        "Content-Type": "application/json",
    }

    cnt = 0
    while True:
        cnt += 1
        try:
            with requests.session() as sess:
                response = sess.post(url, headers=headers, json=data, timeout=120)
            
            # return result
        except BaseException as e:
            if cnt == 5:
                err_msg = f"fail to calling LLM {cnt} times, {e}"
                print(err_msg)
            if cnt == 10:
                return "error"
            continue

        if response.status_code == 200:
            response = response.json()
            if n == 1:
                res = response["choices"][0]["message"]["content"].strip()
            else:
                res = [r["message"]["content"].strip() for r in response["choices"]]
            return res
        
        time.sleep(1)
        return "error"

        
def deepseek_request(model, final_input, temperature, top_p=1, max_tokens=1024, frequency_penalty=0, presence_penalty=0,
                     n=1, stop=None):
    url = "https://api.deepseek.com/chat/completions"
    api_keys = [
        "sk-e76614db92", # 
        # "sk-ed4ba6be5", # 
        # "sk-f2c47931a" # 
    ]

    # ----------------以下不要注释--------------------
    data = {
        'model': model,
        'messages': final_input,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty,
        'n': n,
        'stop': stop
    }

    num_key = len(api_keys)

    key_id = 0
    headers = {
        "Authorization": f"Bearer {api_keys[key_id % num_key]}",
        "Content-Type": "application/json",
    }
    cnt = 0
    while True:
        cnt += 1
        try:
            with requests.session() as sess:
                response = sess.post(url, headers=headers, json=data, timeout=120)
        except BaseException as e:
            if cnt == 3:
                err_msg = f"fail to calling LLM {cnt} times, {e}"
                print(err_msg)
            if cnt == 5:
                return "error"
            continue

        if response.status_code == 200:
            response = response.json()
            if n == 1:
                res = response["choices"][0]["message"]["content"].strip()
            else:
                res = [r["message"]["content"].strip() for r in response["choices"]]
            return res
        elif response.status_code == 400:
            print("Content Exists Risk", )
            return "error"

        
        time.sleep(1)
        return "error"
    

def siliconflow_request_chat(api_key: str, 
                             model: str, 
                             messages: typing.List[dict], 
                             timeout: int = 200, 
                             *, 
                             max_tokens: int = 512, 
                             stop: typing.Union[typing.List[str], None] = None, 
                             temperature: float = 1.0, 
                             top_p: float = 1.0, 
                             top_k: int = 50, 
                             frequency_penalty: float = 0.0, 
                             n: int = 1, 
                             **kargs
                             )-> dict:
    '''
    访问硅基流动平台的对话API。
    api_key: 使用的API_Key
    model: 使用的模型名称
    messages: 符合OpenAI调用格式的消息内容
    timeout: 单次网络访问最大时间限制
    max_tokens: 最大生成词元数
    stop: 停止标记，字符串列表格式
    temperature: 采样温度，取值为[0, 2.0]
    top_p: 采样概率累计，取值为(0, 1.0]
    top_k: 采样选取范围
    frequency_penalty: 频率惩罚，正值会降低模型在一行中重复使用已经频繁出现的词的可能性，取值为[-2.0, 2.0]
    n: 生成样本数量
    '''

    url = "https://api.siliconflow.cn/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
        }

    payload = { # 参数设置
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "stop": stop,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "n": n,
        "response_format": {"type": "text"}
        }

    response = requests.request("POST", url, json=payload, headers=headers, timeout=timeout)
    if response.status_code != 200:
        response.raise_for_status()

    return response.json()