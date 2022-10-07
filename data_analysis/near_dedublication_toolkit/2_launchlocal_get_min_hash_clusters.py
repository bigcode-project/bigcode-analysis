import os

langs = [
    (219.0, 'visual-basic'),
    (259.0, 'fortran'),
    (346.0, 'assembly'),
    (401.0, 'batchfile'),
    (469.0, 'tex'),
    (499.0, 'julia'),
    (557.0, 'powershell'),
    (596.0, 'perl'),
    (600.0, 'cmake'),
    (851.0, 'lua'),
    (980.0, 'haskell'),
    (1228.8, 'makefile'),
    (1331.2, 'sql'),
    (1433.6, 'dockerfile'),
    (2969.6, 'scala'),
    (3276.8, 'rust'),
    (3788.8, 'shell'),
    (6041.6, 'css'),
    (6656.0, 'ruby'),
    (13312.0, 'go'),
    (15360.0, 'c++'),
    (20480.0, 'typescript'),
    (21504.0, 'c'),
    (22528.0, 'c#'),
    (25600.0, 'python'),
    (28672.0, 'markdown'),
    (35840.0, 'html'),
    (36864.0, 'php'),
    (41984.0, 'javascript'),
    (48128.0, 'java')
 ]

script = "text2code_dataset/dataset/postprocessing/near_dedup/2_get_min_hash_clusters"

def run():
    os.system('make build-slim-image')
    os.system('make push-slim-image')

    for i in range(len(langs)-1, -1, -1):
        sz, lang = langs[i]
        print('size: ', sz, ' lang: ', lang)

        #TODO: make it more fine grained
        if sz > 10000:
            mem = 698
        elif sz > 5000:
            mem = 256
        else:
            mem = 128

        command = f'make {script}.launch-slim-nowait SCRIPT_ARGS="{lang}" MORE_JOB_ARGS="--mem {mem} --cpu 8"'

        print(command)
        os.system(command)


if __name__ == "__main__":
    run()

