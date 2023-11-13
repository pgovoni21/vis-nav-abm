from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    set_env_var('CNN_DEPTHS', '1,1')
    set_env_var('CNN_DIMS', '2,4')

    # # # compute 9
    # set_env_var('EA_MOMENTUM', '0.9')
    # set_env_var('EA_STEP_MU', '0.2')
    # for x in range(15):
    #     set_env_var('EA_SAVE_NAME', f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom9_rep{x+5}')
    #     start_EA()

    # # compute8
    # set_env_var('EA_MOMENTUM', '0.9')
    # set_env_var('EA_STEP_MU', '0.10')
    # for x in range(20):
    #     set_env_var('EA_SAVE_NAME', f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss10_mom9_inter_rep{x}')
    #     start_EA()
    # set_env_var('EA_STEP_MU', '0.25')
    # for x in range(20):
    #     set_env_var('EA_SAVE_NAME', f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss25_mom9_inter_rep{x}')
    #     start_EA()

    # # eimmart
    # set_env_var('EA_MOMENTUM', '0.7')
    # set_env_var('EA_STEP_MU', '0.15')
    # for x in range(20):
    #     set_env_var('EA_SAVE_NAME', f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss15_mom7_inter_rep{x}')
    #     start_EA()

    # # compute 4
    # set_env_var('EA_MOMENTUM', '0.7')
    # set_env_var('EA_STEP_MU', '0.2')
    # for x in range(20):
    #     set_env_var('EA_SAVE_NAME', f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom7_rep{x}')
    #     start_EA()

    # # compute 2
    # set_env_var('EA_MOMENTUM', '.8')
    # set_env_var('EA_STEP_MU', '0.2')
    # for x in range(15):
    #     set_env_var('EA_SAVE_NAME', f'singlecorner_exp_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x+5}')
    #     start_EA()

if __name__ == '__main__':

    EA_runner()