import argparse
import multiprocessing
import pathlib
import fasttext

from datasets import load_dataset

from text_extraction import get_text

#adapted from: https://github.com/bigscience-workshop/data-preparation/blob/main/sourcing/
# cc_pseudo_crawl/language_annotation/python_scripts/annotate_langid_crawl.py

COLUMN = "content"

def parseArgs():
    parser = argparse.ArgumentParser(
        description="Identify natural languages in code"
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="HF repo name/path of the dataset.",
    )
    parser.add_argument(
        "save_path",
        default="./data_with_language/",
        type=str,
        help="Path to save the new dataset with language column.",
    )
    parser.add_argument(
        "model_path",
        default= "fasttext_model/lid.176.bin",
        type=str,
        help="Path to fasttext model.",
    )
    args = parser.parse_args()
    return args

def load_fasttext_model(path_fasttext_model):
    return fasttext.load_model(path_fasttext_model)


def get_fasttext_info(line, model_lang_id):
    """The line should be in lower case and without \n in it."""
    pred = model_lang_id.predict(line)
    lang_pred_fasttext_id = pred[0][0].replace("__label__", "")
    score_pred = pred[1][0]
    return lang_pred_fasttext_id, score_pred


def get_all_fasttext_info(document, model_lang_id):
    document = document.lower()
    lang_pred_fasttext_id, score_pred = get_fasttext_info(
        document.replace("\n", " "), model_lang_id
    )
    info = {
        "lang_pred_fasttext_id": lang_pred_fasttext_id,
        "score_pred": score_pred,
        "on_lines": [
            {
                "id_line": id_line,
                "number_caracters_line": len(line),
                "lang_pred_fasttext_id_line": result_fasttext_line[0],
                "score_pred_line": result_fasttext_line[1],
            }
            for id_line, line in enumerate(document.split("\n"))
            for result_fasttext_line in [get_fasttext_info(line, model_lang_id)]
        ],
    }
    return info


def extract_nl_text(example):
        text = get_text(example[COLUMN])
        example["nl_text"] = text
        example["nl_size"] = len(text)
        return example


class FunctionDatasetModifyingDocuments:
    def __init__(self, path_fasttext_model):
        self.path_fasttext_model = path_fasttext_model
        self.model_lang_id = load_fasttext_model(path_fasttext_model)

    def __call__(self, example):
        fasttext_pred = get_all_fasttext_info(
            example["nl_text"], self.model_lang_id
        )
        example["nl_language"] = fasttext_pred["lang_pred_fasttext_id"]
        example["nl_language_score"]  = fasttext_pred["score_pred"]
        return example

    def __reduce__(self):
        return (self.__class__, (self.path_fasttext_model,))


def main():
    args = parseArgs()

    dataset = load_dataset(args.dataset_name)
    print("Loading dataset done")

    func_dataset_modifying_documents = FunctionDatasetModifyingDocuments(
        args.model_path
    )

    dataset = dataset.map(extract_nl_text, num_proc=multiprocessing.cpu_count())

    # Could be improved by allowing multiprocessing with map (currently doesn't work)
    dataset = dataset.map(
        func_dataset_modifying_documents, num_proc=1
    )  # num_proc=cpu_count()
    print("Fasttext done")

    pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(args.save_path)
    print("Shard successfully saved")