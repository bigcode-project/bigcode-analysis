# Profiling of multi head vs multi query attention separately
- `attention_types_imp.py` contains simplistic implementations of different attention layers without normalization, masks and softmax, just matrix multiplications and rearranging of tensors:
    - `MultiHead` is a multi head variant closely following the implementaion in Hugging Face.
    - `MultiQuery` is a multi query variant with dimension order of hidden states as in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) `[sl, bs]`. The reordering of the tensors avoids explicit copies here, however, `bmm` subsequently makes internal copies and  speed suffers. TODO: try with separate tensors for `q`, `k` and `v`.
    - `MultiQuery1` uses the same hidden states order as in HF and one explicit `reshape`. It is the fastest and  is currently ported to HF transformers.
- `profile_attention_types.py` contains code to run timing experiments. Results are in `profile_attention_types.json`.
- `profile_attention_types_visualise.ipynb` contains graphs.
- There is uncertainty about the accuracy times of the profiler. Cpu times, through, decrease slightly in proportion, but still remain significant event for bigger tensors. Around 33% for sequence length of ~2K. However, `MultiQuery1` is the fastest and is ported to HF transformers.

# Profiling of multi head vs multi query attention in HF transformers

[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ocramz/bigcode-analysis/blob/sagemaker/profile_mqa.ipynb)

- Implementaion code is [here](`https://github.com/bigcode-project/transformers/tree/multi_query`)
- `profile_hf_generate.py` contains experiments.
- There are 2 implementations variants of multi query attention controlled by `attention_type` parameter:
    - `AttentionType.MULTI_QUERY` with minimal changes to the code.
    - `AttentionType.MULTI_QUERY_1` with some reordering of dimensions from explorations with @harm-devries and bmm instead of matmul similarly as in `MultiQuery1`.
- `AttentionType.MULTI_QUERY_1` is the fastest, with around 24% speedup:
```
-------------------- attention_type == AttentionType.MULTI_QUERY---------------------
{'get_test_batch': 5.9604644775390625e-05, 'generate_text_batch': 18.453815460205078, 'input_batch_size': 8, 'input_batch_length': 16, 'max_gen_length': 1024, 'num_beams': 1, 'do_sample': False, 'pad_token_id': 50256, 'dtype': torch.int64, 'device': device(type='cuda'), 'cuda_device_name': 'Tesla V100-PCIE-16GB-LS'}
-------------------- attention_type == AttentionType.MULTI_QUERY_1---------------------
{'get_test_batch': 4.172325134277344e-05, 'generate_text_batch': 15.190143346786499, 'input_batch_size': 8, 'input_batch_length': 16, 'max_gen_length': 1024, 'num_beams': 1, 'do_sample': False, 'pad_token_id': 50256, 'dtype': torch.int64, 'device': device(type='cuda'), 'cuda_device_name': 'Tesla V100-PCIE-16GB-LS'}
-------------------- attention_type == AttentionType.MULTI_HEAD---------------------
{'get_test_batch': 5.459785461425781e-05, 'generate_text_batch': 19.78107237815857, 'input_batch_size': 8, 'input_batch_length': 16, 'max_gen_length': 1024, 'num_beams': 1, 'do_sample': False, 'pad_token_id': 50256, 'dtype': torch.int64, 'device': device(type='cuda'), 'cuda_device_name': 'Tesla V100-PCIE-16GB-LS'}
```
