import os

from lang_size import langs

script = "text2code_dataset/dataset/postprocessing/near_dedup/3_group_duplicate_cluster_data"

def run():
    os.system('make build-slim-image')
    os.system('make push-slim-image')

    for i in range(len(langs)-1, -1, -1):
        sz, lang = langs[i]
        print('size: ', sz, ' lang: ', lang)

        if sz > 33000:
            mem = 128
        elif sz > 10000:
            mem = 64
        elif sz > 5000:
            mem = 32
        elif sz > 1000:
             mem = 16
        elif sz > 500:
             mem = 8
        else:
             mem = 4

        command = f'make {script}.launch-slim-nowait SCRIPT_ARGS="{i}" MORE_JOB_ARGS="--mem {mem} --cpu 8 --data snow.code_llm.data:/data --data snow.text_to_sql.transformers_cache:/transformers_cache"'

        print(command)
        os.system(command)


if __name__ == "__main__":
    run()

