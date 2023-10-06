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

    d_range = [
        1,2,8
    ]
    h_range = [
        1,2,8
    ]
    m_types = [
        'fnn', 'gru'
    ]

    for m in m_types:
        for d in d_range:
            for h in h_range:

                d = str(d)
                h = str(h)

                set_env_var('CNN_DIMS', d)
                set_env_var('RNN_HIDDEN_SIZE', h)
                set_env_var('RNN_TYPE', m)

                for x in range(100):
                    set_env_var('EA_SAVE_NAME', f'doublecorner_exp_CNN1{d}_{m.upper()}{h}_rep{x}')
                    start_EA()


if __name__ == '__main__':

    EA_runner()