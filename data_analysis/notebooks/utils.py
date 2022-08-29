import torch
from tqdm import tqdm
import numpy as np

def get_size(text):
    # size of a string in bytes
    return len(text.encode('utf-8'))

def add_size(sample):
    sample["size"] = get_size(sample["content"])
    return sample

def sample_eval_losses(model_all_license, model_safe_license, tokenizer_all, tokenizer_safe, ds, n=2000,  device="cuda"):
    """ compute losses on the first n samples for both models"""
    losses_all = []
    losses_safe = []
    model_all_license.to(device)
    model_safe_license.to(device)
    for i in tqdm(range(n)):
        with torch.no_grad():
            tokens_all = torch.tensor(tokenizer_all(ds[i]["content"], truncation=True)['input_ids'])
            tokens_safe = torch.tensor(tokenizer_safe(ds[i]["content"], truncation=True)['input_ids'])
            
            outputs = model_all_license(tokens_all.to(device), labels=tokens_all.to(device))
            losses_all.append(outputs.loss.item())
            outputs = model_safe_license(tokens_safe.to(device), labels=tokens_safe.to(device))
            losses_safe.append(outputs.loss.item())
            
    return losses_all, losses_safe


def get_embeddings(model, tokenizer, ds, n=200, device="cuda"):
    """get embeddings of n files from the iterable dataset ds
    as the average of token embeddings of the file"""
    embeddings = []
    model.to('cuda')
    for i, example in tqdm(enumerate(ds)):
        with torch.no_grad():
            inputs = torch.tensor(tokenizer(example["content"], truncation=True)['input_ids'])
            outputs = model(inputs.to(device), labels=inputs.to(device), output_hidden_states=True)
            embeddings.append(np.mean(outputs.hidden_states[-1].detach().cpu().numpy(),axis=0))
        if i == n - 1:
            break
    return np.array(embeddings)
