from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


# def generate_env_file(env_data, file_name, save_dir):
#     """Generating a single env file under save_folder with file_name including env_data as env format"""
#     Path(save_dir).mkdir()
#     file_path = Path(save_dir, file_name)
#     with open(file_path, "a") as file:
#         for k, v in env_data.items():
#             file.write(f"{k}={v}\n")


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    # start: EA_SAVE_NAME='stationarypoint_CNN18_CTRNN8_p25e5_sig0p1'

    # set_env_var('EA_GENERATIONS', '500')
    # set_env_var('EA_POPULATION_SIZE', '25')
    # set_env_var('EA_EPISODES', '5')

    # rnn_type_iter = ['fnn','ctrnn','gru']
    # rnn_hidden_iter = [8,16]
    # cnn_dims_iter = [4,8]
    # # cnn_depths_iter = [1,2]
    # init_sigma_iter = [10, '0p1']

    # for l in rnn_type_iter:
    #     for i in rnn_hidden_iter:
    #         for j in cnn_dims_iter:
    #             for k in init_sigma_iter:

    #                 name = f'stationarypoint_CNN1{str(j)}_{l.upper()}{str(i)}_p50e10g500_sig{str(k)}'
    #                 print(name)

    #                 dir = Path(__file__).parent / fr'data/simulation_data/{name}'
    #                 if os.path.isdir(dir):
    #                     print(f'exists, continuing')
    #                     continue

    #                 set_env_var('RNN_HIDDEN_SIZE', str(i))
    #                 set_env_var('CNN_DIMS', str(j))
    #                 set_env_var('CNN_DEPTHS', str(k))
    #                 set_env_var('RNN_TYPE', l)
    #                 set_env_var('EA_SAVE_NAME', name)
    #                 start_EA()


    name = f'stationarypoint_CNN48_FNN8_p25e5g500_sig0p1'    
    print(name)
    set_env_var('CNN_DEPTHS', '4')
    set_env_var('EA_SAVE_NAME', name)
    start_EA()


if __name__ == '__main__':

    EA_runner()