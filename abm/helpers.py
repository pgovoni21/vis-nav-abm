import functools
import time
import os
import linecache
import tracemalloc
from pathlib import Path
import shutil

def timer(func):
    """Print runtime of decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        if run_time > 1e-04:
            print(f"{func.__name__!r} : {run_time:.4f} secs")
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


def folder_trim():

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

    existing_names = [f'singlecorner_exp_CNN16_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x+12}' for x in range(8)]
    new_names = [f'singlecorner_exp_CNN1124_FNN2_p50e20_vis9_PGPE_ss20_mom8_rep{x+12}' for x in range(8)]

    for e,n in zip(existing_names, new_names):
        os.rename(Path(root_dir, e), Path(root_dir, n))


if __name__ == '__main__':

    rename_folders()