import json
import logging

import requests


def get_llama_messages(user_prompt, sys_prompt):
    input_prompt = f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n "
    input_prompt = input_prompt + str(user_prompt) + " [/INST] " + "Rating"
    return input_prompt


def run_llama_eval(
    sys_prompt,
    user_prompt: str,
    hf_token: str,
    api_url: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    logger: logging.Logger = None,
):
    input_prompt = get_llama_messages(user_prompt, sys_prompt)
    data = {
        "inputs": input_prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
        },
    }
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(
        api_url,
        headers=headers,
        data=json.dumps(data),
        auth=("hf", hf_token),
        stream=False,
    )

    if response.status_code == 200:
        response_data = response.json()
        generated_text = response_data["generated_text"]
        return generated_text, True
    else:
        logger.error(f"Request failed with status code: {response.status_code}")
        return "", False
