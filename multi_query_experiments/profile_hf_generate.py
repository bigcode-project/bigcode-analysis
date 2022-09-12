import os
os.environ['TRANSFORMERS_CACHE'] = '/home/toolkit/hf_transformers_cache'

import torch
import time
import transformers

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import AttentionType

print(transformers.__file__)
print(torch.cuda.get_device_name(0))

def get_test_batch(vocab_size, size, length, dtype=torch.int64, device=None):
    #TODO: eliminate special tokens, for now assumes the last one is the only special token
    return {
        'input_ids': torch.randint(0, vocab_size-1, (size, length), dtype=dtype, device=device),
        'attention_mask': torch.ones((size, length), dtype=dtype, device=device)
    }

def generate_text_batch(model, inputs, max_length, num_beams=1, do_sample=False, pad_token_id=50256):
    return model.generate(
        **inputs,  max_length=max_length, num_beams=num_beams, do_sample=do_sample, pad_token_id=pad_token_id
    )

def decode_batch(tokenizer, outputs):
    outputs = outputs.numpy().tolist()
    return [
        tokenizer.decode(output)
        for output in outputs
    ]

def time_generate(
    vocab_size, model, input_batch_size, input_batch_length, max_gen_length,
    num_beams=1, do_sample=False, pad_token_id=50256, dtype=torch.int64, device=None
):
    stats = {}

    t1 = time.time()
    inputs = get_test_batch(vocab_size, input_batch_size, input_batch_length, dtype, device)
    stats['get_test_batch'] = time.time() - t1

    t1 = time.time()
    outputs = generate_text_batch(
        model, inputs, max_gen_length, num_beams=num_beams, do_sample=do_sample, pad_token_id=pad_token_id
    )
    stats['generate_text_batch'] = time.time() - t1
    stats['input_batch_size'] = input_batch_size
    stats['input_batch_length'] = input_batch_length
    stats['max_gen_length'] = max_gen_length
    stats['num_beams'] = num_beams
    stats['do_sample'] = do_sample
    stats['pad_token_id'] = pad_token_id
    stats['dtype'] = dtype
    stats['device'] = device
    stats['cuda_device_name'] = torch.cuda.get_device_name(0)

    return inputs, outputs, stats

def profile(attention_type):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_layer=24,
        n_embd=1024,
        n_head=16,
        n_positions=2048,
        #n_ctx=tokenizer.model_max_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_type=attention_type,
        print_details=False
    )
    model = GPT2LMHeadModel(config)
    model = model.to(torch.device('cuda'))
    
    inputs = get_test_batch(tokenizer.vocab_size, 1, 4, device=torch.device('cuda'))
    
    print(f'-------------------- attention_type == {attention_type}---------------------')
    
    inputs, outputs, stats = time_generate(tokenizer.vocab_size, model, 8, 16, 1024, device=torch.device('cuda'))
    print(stats)
    
# warm up    
profile(AttentionType.MULTI_QUERY)

profile(AttentionType.MULTI_QUERY)
profile(AttentionType.MULTI_QUERY_1)
profile(AttentionType.MULTI_HEAD)
