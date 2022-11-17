import os

from lang_size import langs

script = "text2code_dataset/dataset/postprocessing/near_dedup/2_get_min_hash_clusters"

def run():
    os.system('make build-slim-image')
    os.system('make push-slim-image')

    for i in range(len(langs)-1, -1, -1):
        sz, lang = langs[i]
        print('size: ', sz, ' lang: ', lang)

        #TODO: make it more fine grained
        if sz > 33000:
            mem = 698
        elif sz > 10000:
            mem = 256
        elif sz > 5000:
            mem = 128
        elif sz > 1000:
             mem = 128
        elif sz > 500:
             mem = 64
        else:
             mem = 16
        command = f'make {script}.launch-slim-nowait SCRIPT_ARGS="{i}" MORE_JOB_ARGS="--mem {mem} --cpu 8"'

        print(command)
        os.system(command)


if __name__ == "__main__":
    run()

