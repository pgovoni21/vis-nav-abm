import functools
import time
import os
import linecache
import tracemalloc
from pathlib import Path
import shutil
import dotenv as de

def timer(func):
    """Print runtime of decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        if run_time > 1e-05:
            print(f"{func.__name__!r} : {run_time:.5f} secs")
        return value
    return wrapper_timer
    return None

def debug(func):
    """Print function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


def display_top_mem_users(snapshot, key_type='lineno', limit=3):

    # use with:
    # tracemalloc.start() --> to initiate
    # display_top_mem_users(tracemalloc.take_snapshot()) --> to print memory usage

    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def trim_folders():

    root_dir = Path(__file__).parent / fr'data/simulation_data'
    for file in os.listdir(root_dir):

        dir = Path(root_dir, file)
        print(dir)

        for g in range(1000):
            if os.path.isdir(Path(dir,fr'gen{g}')):
                for file in os.listdir(Path(dir,fr'gen{g}')):
                    if file.startswith('NN0'):
                        if not file.endswith('png'):
                            shutil.move(fr'{dir}/gen{g}/{file}/NN_pickle.bin', fr'{dir}/gen{g}_NN0.bin')
                shutil.rmtree(fr'{dir}/gen{g}')

            # else: 
            #     print(f'failed for: {dir,g}')



def rename_folders():

    root_dir = Path(__file__).parent / fr'data/simulation_data'

    # existing_names = [f'singlecorner_exp_CNN16_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x+12}' for x in range(8)]
    # new_names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x+12}' for x in range(8)]

    # for e,n in zip(existing_names, new_names):
    #     os.rename(Path(root_dir, e), Path(root_dir, n))

    word_to_change = 'vis10_PGPE_ss20_mom8_head44'
    for old_name in os.listdir(root_dir):
        if word_to_change in old_name:
            print(word_to_change)
            new_name = old_name.replace(word_to_change, 'vis8_PGPE_ss20_mom8_head44')
            os.rename(Path(root_dir, old_name), Path(root_dir, new_name))

def set_env_var(env_path, key, val):
    de.set_key(env_path, str(key), str(val))

def modify_env_files():

    root_dir = Path(__file__).parent / fr'data/simulation_data'

    # existing_names = [f'singlecorner_exp_CNN16_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x+12}' for x in range(8)]
    # new_names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x+12}' for x in range(8)]

    # for e,n in zip(existing_names, new_names):
    #     os.rename(Path(root_dir, e), Path(root_dir, n))

    # word_to_change = 'vis10_PGPE_ss20_mom8_head44'
    # for old_name in os.listdir(root_dir):
    #     if word_to_change in old_name:
    #         print(word_to_change)
    #         new_name = old_name.replace(word_to_change, 'vis8_PGPE_ss20_mom8_head44')
    #         os.rename(Path(root_dir, old_name), Path(root_dir, new_name))

    for name in os.listdir(root_dir):
        if name.startswith('sc_') and not name.endswith('png'):
            # print(name)

            env_path = fr'{root_dir}/{name}/.env'
            envconf = de.dotenv_values(env_path)

            set_env_var(env_path, 'PERCEP_ANGLE_NOISE_STD', '0')
            set_env_var(env_path, 'ACTION_NOISE_STD', '0')

            if 'SENSORY_NOISE_STD' in envconf:
                set_env_var(env_path, 'PERCEP_DIST_NOISE_STD', envconf['SENSORY_NOISE_STD'])
            else:
                set_env_var(env_path, 'PERCEP_DIST_NOISE_STD', '0')

            if 'ENV_SIZE' not in envconf:
                set_env_var(env_path, 'ENV_SIZE', '(1000,1000)')

            if 'RESOURCE_UNITS' not in envconf:
                set_env_var(env_path, 'RESOURCE_UNITS', '(1,1)')

            if 'RESOURCE_QUALITY' not in envconf:
                set_env_var(env_path, 'RESOURCE_QUALITY', '(1,1)')

            if 'RESOURCE_POS' not in envconf:
                set_env_var(env_path, 'RESOURCE_POS', '(400,400)')

            if 'VIS_TRANSFORM' not in envconf:
                set_env_var(env_path, 'VIS_TRANSFORM', '')


if __name__ == '__main__':

    # trim_folders()
    # rename_folders()
    modify_env_files()