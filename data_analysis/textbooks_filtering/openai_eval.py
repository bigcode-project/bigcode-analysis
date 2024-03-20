import logging
import time

import openai


def get_openai_messages(user_prompt, sys_prompt):
    messages = [{"role": "system", "content": sys_prompt}]
    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )
    return messages


def run_openai_eval(
    sys_prompt,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model: str = "gpt-3.5-turbo",
    logger: logging.Logger = None,
):
    logging.basicConfig(level=logging.INFO)
    MAX_OPENAI_API_RETRY = 6
    for i in range(MAX_OPENAI_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,  # "gpt-4",
                messages=get_openai_messages(user_prompt, sys_prompt),
                temperature=temperature,  # TODO: figure out which temperature is best for evaluation
                top_p=top_p,
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            return content, True
        except Exception as e:
            if logger is not None:
                logger.error(e)
            time.sleep(5)  # TODO see how low this can go

    if logger is not None:
        logger.error(f"Failed after {MAX_OPENAI_API_RETRY} retries.")
    return None, False