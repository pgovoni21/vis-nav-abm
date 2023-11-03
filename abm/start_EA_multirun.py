from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    set_env_var('EA_MOMENTUM', '0.6')
    for x in range(5):
        set_env_var('EA_SAVE_NAME', f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom6_rep{x}')
        start_EA()
    set_env_var('EA_MOMENTUM', '1')
    for x in range(5):
        set_env_var('EA_SAVE_NAME', f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom10_rep{x}')
        start_EA()


if __name__ == '__main__':

    EA_runner()