# Some chunks is this code are adapted from https://github.com/huggingface/h4/tree/scripts/evaluation
import json
import logging
import os
import time
from argparse import ArgumentParser
from collections import Counter
from multiprocessing.pool import Pool
from time import gmtime, strftime

import openai
from datasets import load_dataset
from llama_eval import run_llama_eval
from openai_eval import run_openai_eval
from utils import parse_score
from tqdm.auto import tqdm
from transformers import set_seed


REQ_OPENAI_TIME_GAP = {"gpt-4": 6, "gpt-3.5-turbo": 6}
SYSTEM_PROMPT = "You are a helpful and precise assistant for checking the educational quality of code."
PROMPT_HEADER = """Please act as an impartial judge and evaluate the educational value of the code file displayed below for someone just starting to learn coding concepts. Your evaluation should prioritize clarity and simplicity to ensure the code is easily digestible for a beginner. \
Be as objective as possible. You must first rate the code file on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]", then provide a short explanation of the rating.\n\nCode file:\n\n"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for random number generator",
    )
    parser.add_argument(
        "--model_type",
        default="openai",
        type=str,
        help="Model type to use for annotation, can be 'openai' or 'llama'",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-3.5-turbo",
        type=str,
        help="Model name to use, use official OpenAI model name or 'llama' for llama-70b-chat",
    )
    parser.add_argument(
        "--n_samples",
        default=500,
        type=int,
        help="Number of samples to annotate",
    )
    parser.add_argument(
        "--output_path",
        default="/fsx/loubna/code/data_v2/analysis/openai/eval_openai_results.json",
        type=str,
        help="Output path for annotations",
    )
    parser.add_argument(
        "--log_file",
        default="/fsx/loubna/code/data_v2/analysis/openai/eval_openai.log",
        type=str,
        help="Log file path",
    )
    return parser.parse_args()


def build_prompt(i, ds, header=PROMPT_HEADER):
    input = header
    code = ds[i]
    prompt = f"{input}{code['content']}"
    return prompt


def run_eval(inputs):
    input_text, model_name, sleep_time, hf_token, api_url = inputs
    # To avoid the rate limit
    if model_name == "llama-70b-chat":
        # parameters taken from https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI
        completion, did_run = run_llama_eval(
            SYSTEM_PROMPT,
            input_text,
            hf_token,
            api_url,
            max_tokens=128,
            temperature=0.9,
            top_p=0.6,
            logger=logger,
        )
    else:
        # OpenAI models
        assert model_name in [
            "gpt-4",
            "gpt-3.5-turbo",
        ], "Only gpt-4 and gpt-3.5-turbo are accepted"
        completion, did_run = run_openai_eval(
            SYSTEM_PROMPT,
            input_text,
            max_tokens=128,
            temperature=0.7,
            top_p=0.95,
            model=model_name,
            logger=logger,
        )
    if not did_run:
        print("Skipping failed run")
        return {
            "prompt": input_text,
            "completion": "",
            "review_id": "",
            "review_model": model_name,
            "eval_prompt_header": PROMPT_HEADER,
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.95,
            },
        }

    review_dict = {
        "completion": completion,
        "review_model": model_name,
        "prompt": input_text,
        "eval_prompt_header": PROMPT_HEADER,
        "generation_config": {
            "temperature": 0.7,
            "top_p": 0.95,
        },
    }

    # Optional sleep
    time.sleep(sleep_time)
    if sleep_time > 0:
        logger.info(f"Sleeping for {sleep_time} seconds to avoid rate limit.")

    return review_dict


# if mainname
if __name__ == "__main__":
    args = get_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    set_seed(args.seed)

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    hf_token = os.environ.get("HF_LLAMA_TOKEN")
    api_url = os.environ.get("LLAMA_API_URL")

    openai_model = args.model_type == "openai"
    model_name = args.model_name
    sleep_time = 0 if not openai_model else REQ_OPENAI_TIME_GAP[model_name]

    # Load dataset
    ds = load_dataset("bigcode/the-stack-smol", split="train", data_dir="data/python")
    ds_medium = ds.filter(lambda x: 600 <= x["size"] <= 8000)
    logger.info(
        f"Loaded {len(ds_medium)} medium-sized (600-6000 characters) code files."
    )

    # build prompts
    prompts = [build_prompt(i, ds_medium, header=PROMPT_HEADER) for i in range(args.n_samples)]
    logger.info(
        f"Built {len(prompts)} prompts for {model_name} model.\n{'='*50} Sample prompt {'='*50}\n{prompts[0]}"
    )

    # Run eval
    inputs = [(prompt, model_name, sleep_time, hf_token, api_url) for prompt in prompts]
    logger.info(f"Running eval for {len(inputs)} prompts with {model_name} model.")
    with Pool(12) as pool:
        review_jsons = list(tqdm(pool.imap(run_eval, inputs), total=len(prompts)))

    # add score to each value in dict
    reviews_with_scores = [
        {**review_json, "score": parse_score(review_json["completion"], logger)}
        for review_json in review_jsons
    ]
    scores = [review["score"] for review in reviews_with_scores]
    logger.info(f"Distribution of scores is: {Counter(scores)}")

    dumped = json.dumps(reviews_with_scores, indent=4, sort_keys=True, default=str)
    with open(args.output_path, "w") as output_review_file:
        output_review_file.write(dumped)
        logger.info(f"🚀 All done! Completions saved to {args.output_path}")
