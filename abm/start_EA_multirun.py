from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    set_env_var('EA_POPULATION_SIZE', '50')
    set_env_var('EA_EPISODES', '20')
    set_env_var('EA_EST_METHOD', 'mean')

    set_env_var('CNN_DEPTHS', '1,1')
    set_env_var('CNN_DIMS', '2,4')
    set_env_var('RNN_HIDDEN_SIZE', '2')
    set_env_var('RNN_TYPE', 'fnn')

    set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    for x in range(20):
        set_env_var('EA_SAVE_NAME', f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_nodist_rep{x}')
        start_EA()


if __name__ == '__main__':

    EA_runner()