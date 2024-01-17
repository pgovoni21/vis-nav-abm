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



def rename_files():

    root_dir = Path(__file__).parent / fr'data/simulation_data'
    # root_dir = Path(__file__).parent / fr'data/simulation_data/traj_matrices'

    word_to_change = 'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist'
    for old_name in os.listdir(root_dir):
        if word_to_change in old_name:
            print(old_name)
            new_name = old_name.replace(word_to_change, 'lmdist_n')
            os.rename(Path(root_dir, old_name), Path(root_dir, new_name))



def modify_env_files():

    root_dir = Path(__file__).parent / fr'data/simulation_data'

    for name in os.listdir(root_dir):
        if name.startswith('sc_') and not name.endswith('png'):

            env_path = fr'{root_dir}/{name}/.env'
            envconf = de.dotenv_values(env_path)

            # de.set_key(env_path, 'PERCEP_LM_RADIUS_NOISE_STD', '0')

            # if 'PERCEP_LM_NOISE_STD' in envconf:
            #     print(name)
            # if 'LM_DIST_NOISE_STD' in envconf:
            #     print(name)
            # if envconf['LM_DIST_NOISE_STD'] != '0':
            #     print(name)

            if 'PERCEP_LM_DIST_NOISE_STD' in envconf:
                val = envconf['PERCEP_LM_DIST_NOISE_STD']
                de.set_key(env_path, 'LM_DIST_NOISE_STD', val)
                de.unset_key(env_path, 'PERCEP_LM_DIST_NOISE_STD')
            if 'PERCEP_LM_RADIUS_NOISE_STD' in envconf:
                val = envconf['PERCEP_LM_RADIUS_NOISE_STD']
                de.set_key(env_path, 'LM_RADIUS_NOISE_STD', val)
                de.unset_key(env_path, 'PERCEP_LM_RADIUS_NOISE_STD')
            if 'PERCEP_LM_ANGLE_NOISE_STD' in envconf:
                val = envconf['PERCEP_LM_ANGLE_NOISE_STD']
                de.set_key(env_path, 'LM_ANGLE_NOISE_STD', val)
                de.unset_key(env_path, 'PERCEP_LM_ANGLE_NOISE_STD')

            # if 'SENSORY_NOISE_STD' in envconf:
            #     de.unset_key(env_path, 'SENSORY_NOISE_STD')

            # if 'PERCEP_ANGLE_NOISE' in envconf:
            #     de.unset_key(env_path, 'PERCEP_ANGLE_NOISE')

            # if 'ACTION_NOISE' in envconf:
            #     de.unset_key(env_path, 'ACTION_NOISE')



if __name__ == '__main__':

    # trim_folders()
    # rename_files()
    modify_env_files()