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

    vis_res_iter = ['8','32']
    cnn_deps_iter = ['1','2']
    cnn_dims_iter = ['2','8']
    rnn_hidden_iter = ['2','8']

    for i in vis_res_iter:
        for j in rnn_hidden_iter:
            for k in cnn_dims_iter:
                for l in cnn_deps_iter:

                    name = f'doublepoint_CNN{l}{k}_GRU{j}_p25e5g1000_sig0p1_vis{i}'
                    print(name)

                    # dir = Path(__file__).parent / fr'data/simulation_data/{name}'
                    # if os.path.isdir(dir):
                    #     print(f'exists, continuing')
                    #     continue

                    set_env_var('VISUAL_FIELD_RESOLUTION', i)
                    set_env_var('RNN_HIDDEN_SIZE', j)
                    set_env_var('CNN_DIMS', k)
                    set_env_var('CNN_DEPTHS', l)
                    set_env_var('EA_SAVE_NAME', name)
                    start_EA()

    set_env_var('VISUAL_FIELD_RESOLUTION', '32')
    set_env_var('RNN_HIDDEN_SIZE', '8')
    set_env_var('CNN_DEPTHS', '1,1')
    set_env_var('CNN_DIMS', '2,8')
    set_env_var('EA_SAVE_NAME', f'doublepoint_CNN1128_GRU8_p25e5g1000_sig0p1_vis32')
    start_EA()
    set_env_var('CNN_DEPTHS', '1,1,1')
    set_env_var('CNN_DIMS', '2,4,8')
    set_env_var('EA_SAVE_NAME', f'doublepoint_CNN111248_GRU8_p25e5g1000_sig0p1_vis32')
    start_EA()

    set_env_var('CNN_DEPTHS', '1,1')
    set_env_var('CNN_DIMS', '2,4')
    set_env_var('EA_SAVE_NAME', f'doublepoint_CNN1124_GRU8_p25e5g1000_sig0p1_vis32')
    start_EA()
    set_env_var('CNN_DEPTHS', '1,1,1')
    set_env_var('CNN_DIMS', '2,4,6')
    set_env_var('EA_SAVE_NAME', f'doublepoint_CNN111246_GRU8_p25e5g1000_sig0p1_vis32')
    start_EA()


if __name__ == '__main__':

    EA_runner()