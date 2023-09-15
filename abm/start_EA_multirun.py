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
    
    set_env_var('CNN_DEPTHS', '1,1')
    set_env_var('CNN_DIMS', '2,8')

    set_env_var('RNN_HIDDEN_SIZE', '2')


    set_env_var('RNN_INPUT_OTHER_SIZE', '0')
    set_env_var('RNN_TYPE', 'gru')
    for x in range(20):
        set_env_var('EA_SAVE_NAME', f'symdoublepoint_CNN1128_GRU2_other0_rep{x}')
        start_EA()

    set_env_var('RNN_TYPE', 'fnn')
    for x in range(20):
        set_env_var('EA_SAVE_NAME', f'symdoublepoint_CNN1128_FNN2_other0_rep{x}')
        start_EA()


if __name__ == '__main__':

    EA_runner()