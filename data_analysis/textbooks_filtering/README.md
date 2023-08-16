# Filtering of The Stack with an LLM [WIP]
In this section we describe teh filtering of the Python subset of The Stack following the approach used in [Textbooks are all you need paper]().
The filtering is done in two steps:
1 - Annotate 100k files from the Python subset of The Stack using GPT4/LLaMa to find if a file has educational value for beginners or not.
2 - Use the annotations to train a classifier to predict if a file has educational value for beginners or not based on its embedding.

### Annotating The Stack
We first test LLaMa-70B-Chat. We use the following prompt:
```python
prompt = """Please act as an impartial judge and evaluate the educational value of the code file displayed below for someone just starting to learn coding concepts. Your evaluation should prioritize clarity and simplicity to ensure the code is easily digestible for a beginner. \
Be as objective as possible. You must first rate the code file on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]", then provide a short explanation of the rating.\n\nCode file:\n\n"""
# append code file
```
We now only analyze code file with more than 600 charcaters and less than 6k characters.
After setting the variable `HF_TOKEN`, `LLAMA_API_URL` and `OPENAI_API_KEY` you can run annotaions. For example, you can use LLaMa-70B-Chat to annotate the files using the following command:
```bash
python main.py --model_type llama --model_name llama-70b-chat --n_samples 200 --output_path ./llama_200_samples.json
python main.py --model_type openai --model_name gpt4 --n_samples 600 --output_path ./gpt4_600_samples.json
python main.py --model_type openai --model_name gpt-3.5-turbo --n_samples 10 --output_path ./chatgpt_10_samples.json
...
You can find some analysis of results in `analyze_results.ipynb` notebook including teh distribution of scores on 600 python files below:

<img src="https://huggingface.co/datasets/loubnabnl/repo-images/resolve/main/llms_stack.png" alt="llms_stack" width="200" height="400"/>