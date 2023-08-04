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

    rnn_type_iter = ['fnn','ctrnn','gru']
    cnn_dims_iter = ['2','4','8']
    rnn_hidden_iter = ['2','8']

    for i in rnn_type_iter:
        for j in rnn_hidden_iter:
            for k in cnn_dims_iter:

                name = f'crosscorner_CNN1{k}_{i.upper()}{str(j)}_p25e5g500_sig0p1'
                print(name)

                # dir = Path(__file__).parent / fr'data/simulation_data/{name}'
                # if os.path.isdir(dir):
                #     print(f'exists, continuing')
                #     continue

                set_env_var('RNN_TYPE', i)
                set_env_var('RNN_HIDDEN_SIZE', j)
                set_env_var('CNN_DIMS', k)
                set_env_var('EA_SAVE_NAME', name)
                start_EA()


if __name__ == '__main__':

    EA_runner()