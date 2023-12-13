from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    # # # gadus
    for x in range(20):
        set_env_var('CNN_DEPTHS', '1')
        set_env_var('CNN_DIMS','4')
        set_env_var('AGENT_FOV','.4')
        set_env_var('VIS_TRANSFORM','WF')
        set_env_var('SENSORY_NOISE_STD','.2')
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep{x}')
        start_EA()

    # # # fish
    for x in range(20):
        set_env_var('CNN_DEPTHS', '1')
        set_env_var('CNN_DIMS','4')
        set_env_var('AGENT_FOV','.4')
        set_env_var('VIS_TRANSFORM','WF')
        set_env_var('SENSORY_NOISE_STD','0')
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{x}')
        start_EA()

    # # # compute 2
    # for x in range(20):
    #     set_env_var('CNN_DEPTHS', '1')
    #     set_env_var('CNN_DIMS','4')
    #     set_env_var('AGENT_FOV','.4')
    #     set_env_var('VISUAL_FIELD_RESOLUTION','6')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}')
    #     start_EA()

    # # # compute4
    # for x in range(18):
    #     set_env_var('CNN_DEPTHS', '1')
    #     set_env_var('CNN_DIMS','4')
    #     set_env_var('AGENT_FOV','.4')
    #     set_env_var('VISUAL_FIELD_RESOLUTION','16')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x+2}')
    #     start_EA()

    # # # michaelis
    # for x in range(20):
    #     set_env_var('CNN_DEPTHS', '1')
    #     set_env_var('CNN_DIMS','4')
    #     set_env_var('AGENT_FOV','.4')
    #     set_env_var('VISUAL_FIELD_RESOLUTION','14')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}')
    #     start_EA()

    # # # menten
    # for x in range(20):
    #     set_env_var('CNN_DEPTHS', '1')
    #     set_env_var('CNN_DIMS','7')
    #     set_env_var('AGENT_FOV','.4')
    #     set_env_var('VISUAL_FIELD_RESOLUTION','8')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    #     start_EA()


if __name__ == '__main__':

    EA_runner()