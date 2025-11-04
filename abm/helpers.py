import functools
import time
import os
import linecache
import tracemalloc
from pathlib import Path
import shutil
import dotenv as de
import pickle

# Global dictionary to store accumulated runtimes
accumulated_runtimes = {}

def timer(func):
    """Print runtime of decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3

        # Update accumulated runtime
        if func.__name__ not in accumulated_runtimes:
            accumulated_runtimes[func.__name__] = 0
        accumulated_runtimes[func.__name__] += run_time

        print(accumulated_runtimes)
        # if run_time > 1e-05:
        #     print(f"{func.__name__!r} : {run_time:.5f} secs")

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

        if 'SinitAg100' not in file:
            continue

        if 'multi' in file:
            continue

        dir = Path(root_dir, file)
        # print(dir)

        # for g in range(1000):
        #     if os.path.isdir(Path(dir,fr'gen{g}')):
        #         for file in os.listdir(Path(dir,fr'gen{g}')):
        #             if file.startswith('NN0'):
        #                 if not file.endswith('png'):
        #                     shutil.move(fr'{dir}/gen{g}/{file}/NN_pickle.bin', fr'{dir}/gen{g}_NN0.bin')
        #         shutil.rmtree(fr'{dir}/gen{g}')

        #     # else: 
        #     #     print(f'failed for: {dir,g}')

        for file in os.listdir(dir):
            if file.startswith('gen'):
                file_spaces = file[3:].replace('_',' ')
                gen_num = [int(s) for s in file_spaces.split() if s.isdigit()][0]
                if gen_num > 999:
                    print(Path(dir, file))
                    shutil.rmtree(Path(dir, file))



def rename_files():

    root_dir = Path(__file__).parent / fr'data/simulation_data'
    # root_dir = Path(__file__).parent / fr'data/simulation_data/traj_matrices'

    count = 0
    string = 'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_SinitAg100_rep'
    for name in os.listdir(root_dir):
        if string in name:
            count += 1
            print(name)
            new_name = name.replace(string,'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_SinitAg100_rep')
            # print(name, new_name)
            # shutil.rmtree(Path(root_dir, name))
            os.rename(Path(root_dir, name), Path(root_dir, new_name))

            # env_path = fr'{root_dir}/{name}/.env'
            # envconf = de.dotenv_values(env_path)
            # print(name, envconf['N'], envconf['N_RAND'])
    print(count)


def rename_file_in_folder():

    root_dir = Path(__file__).parent / fr'data/simulation_data'
    # root_dir = Path(__file__).parent / fr'data/simulation_data/traj_matrices'

    log = []
    for folder_name in os.listdir(root_dir):
        if 'sc_N' in folder_name and not folder_name.endswith('png'):
            files = list(os.listdir(root_dir/folder_name))

            for file_name in os.listdir(root_dir/folder_name):
                if 'val_matrix_best_nosocial_perturb.bin' in file_name:
                    # print(folder_name)
                # if 'val_result' in file_name and 'cen' not in file_name:
                    # print(folder_name, file_name)
                    log.append(folder_name)
                    continue
                    # os.rename(Path(root_dir, folder_name, file_name), Path(root_dir, folder_name, 'val_results_cen.txt'))

    # log.sort()
    # for n in log:
    # #     if 'ghost' in n:
    #     print(n)



def modify_env_files():

    root_dir = Path(__file__).parent / fr'data/simulation_data'

    count = 0
    for name in os.listdir(root_dir):
        if 'SinitAg100' in name and 'collinput' not in name and 'multi' not in name and not name.endswith('png'):
        # if name.startswith('nowall') and not name.endswith('png'):
        
            env_path = fr'{root_dir}/{name}/.env'
            envconf = de.dotenv_values(env_path)

            if envconf['EA_GENERATIONS'] == '2500':
                print(name)
            #     count += 1
            
            # if 'NRW3' in name and envconf['N_RAND'] != '3':
            #     print(name)

            # if 'NRW3' in name and envconf['N_RAND'] != '3':
            #     print(name, envconf['N_RAND'])

            # de.set_key(env_path, 'EA_GENERATIONS', '2500')

            # if 'SOCIAL_INIT_RANGE' in envconf.keys():
            #     print(name, envconf['SOCIAL_INIT_RANGE'])

            # if 'SinitRes' in name:
            #     # print(name, envconf['SOCIAL_INIT_TYPE'])

            #     de.set_key(env_path, 'SOCIAL_INIT_TYPE', 'res')


            # if envconf['SIM_TYPE'] == 'walls, social-RW':
            #     # print(name)
            #     de.set_key(env_path, 'SIM_TYPE', 'walls, social')

            # string = 'NRW0_ND5'
            # if string in name:
            #     print(name, envconf['N_RAND'])

                # de.set_key(env_path, 'N_RAND', '0')





            # if envconf['SIM_TYPE'] not in type_list:
            #     type_list.append(envconf['SIM_TYPE'])
            #     print(envconf['SIM_TYPE'])

            # de.set_key(env_path, 'MISC_WEIGHT', '0.25')

            # # if 'seed1' in name and 'mlWF' in name:
            # # if 'seed2' in name and 'sWF' in name:
            # # if 'seed2' in name and '24' in name:
            # if 'seed4' in name and 'dist' not in name and '24' not in name:

            #     env_path = fr'{root_dir}/{name}/.env'
            #     envconf = de.dotenv_values(env_path)

            #     print(name)
            #     # print(envconf["EA_START_SEED"])
            #     print(envconf["SIM_TYPE"])
            #     # de.set_key(env_path, 'SIM_TYPE', 'walls')

            # de.set_key(env_path, 'PERCEP_LM_RADIUS_NOISE_STD', '0')

            # if float(envconf["RADIUS_LANDMARK"]) > 0:
            #     print(float(envconf["RADIUS_LANDMARK"]), name)
            #     de.set_key(env_path, 'RADIUS_LANDMARK', '0')

            # if 'PERCEP_LM_NOISE_STD' in envconf:
            #     print(name)
            # if 'LM_DIST_NOISE_STD' in envconf:
            #     print(name)
            # if envconf['LM_DIST_NOISE_STD'] != '0':
            #     print(name)

            # if 'lm' in name:
            #     continue

            # if 'LM_DIST_NOISE_STD' not in envconf:
            #     print(name, envconf['LM_DIST_NOISE_STD'])
            #     val = envconf['LM_DIST_NOISE_STD']
            #     # de.set_key(env_path, 'LM_DIST_NOISE_STD', val)
            #     # de.unset_key(env_path, 'PERCEP_LM_DIST_NOISE_STD')

            # if 'SIM_TYPE' not in envconf:
            #     print(name)
            #     de.set_key(env_path, 'SIM_TYPE', 'walls')
    # print(count)


def modify_pickled_files():

    root_dir = Path(__file__).parent / fr'data/simulation_data'
    EA_save_name = 'nowall_N5_CNN14_FNN16_vis8_rep0'
    file_path = Path(root_dir, EA_save_name) / 'fitness_spread_per_generation.bin'
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    
    print(file)


if __name__ == '__main__':

    # trim_folders()
    # rename_files()
    rename_file_in_folder()
    # modify_env_files()
    # modify_pickled_files()