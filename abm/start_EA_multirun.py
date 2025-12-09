from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    # for x in range(12):
    #     set_env_var('N', '4')
    #     set_env_var('N_RAND','1')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('CNN_DIMS', '4')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('EA_SAVE_NAME', f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x+48}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('N', '4')
    #     set_env_var('N_RAND','1')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('CNN_DIMS', '4')
    #     set_env_var('EA_START_SEED', '10000')
    #     set_env_var('EA_SAVE_NAME', f'sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep{x+60}')
    #     start_EA()

    # for x in range(11):
    #     set_env_var('N', '5')
    #     set_env_var('N_RAND','2')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('CNN_DIMS', '4')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('EA_SAVE_NAME', f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x+49}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('N', '5')
    #     set_env_var('N_RAND','2')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('CNN_DIMS', '4')
    #     set_env_var('EA_START_SEED', '10000')
    #     set_env_var('EA_SAVE_NAME', f'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep{x+60}')
    #     start_EA()

    # for x in range(13):
    #     set_env_var('N', '6')
    #     set_env_var('N_RAND','3')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('CNN_DIMS', '4')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('EA_SAVE_NAME', f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x+47}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('N', '6')
    #     set_env_var('N_RAND','3')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('CNN_DIMS', '4')
    #     set_env_var('EA_START_SEED', '10000')
    #     set_env_var('EA_SAVE_NAME', f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x+60}')
    #     start_EA()



    # for x in range(4):
    #     set_env_var('N', '6')
    #     set_env_var('SOCIAL_INIT_TYPE', '')
    #     # set_env_var('SOCIAL_INIT_RANGE', '100')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('AGENT_COLLIDE', '1')
    #     set_env_var('EA_GENERATIONS', '2500')
    #     set_env_var('EA_SAVE_NAME', f'sc_N6multi_CNN14_FNN16_vis8_rep{x+1}')
    #     start_EA(EA_type='multi')

    # for x in range(4):
    #     set_env_var('N', '6')
    #     set_env_var('SOCIAL_INIT_TYPE', 'ag')
    #     set_env_var('SOCIAL_INIT_RANGE', '100')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('AGENT_COLLIDE', '1')
    #     set_env_var('EA_GENERATIONS', '2500')
    #     set_env_var('EA_SAVE_NAME', f'sc_N6multi_CNN14_FNN16_vis8_SinitAg100_rep{x+1}')
    #     start_EA(EA_type='multi')

    # for x in range(4):
    #     set_env_var('N', '6')
    #     set_env_var('SOCIAL_INIT_TYPE', '')
    #     # set_env_var('SOCIAL_INIT_RANGE', '100')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('AGENT_COLLIDE', '0')
    #     set_env_var('EA_GENERATIONS', '2500')
    #     set_env_var('EA_SAVE_NAME', f'sc_N6multi_CNN14_FNN16_vis8_nocoll_rep{x+1}')
    #     start_EA(EA_type='multi')


    # for x in range(5):
    #     set_env_var('N', '6')
    #     set_env_var('SOCIAL_INIT_TYPE', '')
    #     # set_env_var('SOCIAL_INIT_RANGE', '100')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('AGENT_COLLIDE', '1')
    #     set_env_var('EA_GENERATIONS', '2500')
    #     set_env_var('EA_SAVE_NAME', f'sc_N6multi_leaderIS_CNN14_FNN16_vis8_rep{x}')
    #     start_EA(EA_type='multi_leader')

    # for x in range(5):
    #     set_env_var('N', '6')
    #     set_env_var('SOCIAL_INIT_TYPE', 'ag')
    #     set_env_var('SOCIAL_INIT_RANGE', '100')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('AGENT_COLLIDE', '1')
    #     set_env_var('EA_GENERATIONS', '2500')
    #     set_env_var('EA_SAVE_NAME', f'sc_N6multi_leaderIS_CNN14_FNN16_vis8_SinitAg100_rep{x}')
    #     start_EA(EA_type='multi_leader')

    # for x in range(5):
    #     set_env_var('N', '6')
    #     set_env_var('SOCIAL_INIT_TYPE', '')
    #     # set_env_var('SOCIAL_INIT_RANGE', '100')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('AGENT_COLLIDE', '0')
    #     set_env_var('EA_GENERATIONS', '2500')
    #     set_env_var('EA_SAVE_NAME', f'sc_N6multi_leaderIS_CNN14_FNN16_vis8_nocoll_rep{x}')
    #     start_EA(EA_type='multi_leader')


    # for x in range(3):
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '32')
    #     set_env_var('AGENT_FOV', '0.97')
    #     set_env_var('RNN_HIDDEN_SIZE', '64')
    #     set_env_var('RNN_TYPE', 'fnn2')
    #     set_env_var('CNN_DIMS', '8')
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'silu')
    #     set_env_var('EA_START_SEED', '1000')
    #     set_env_var('EA_GENERATIONS', '10000')
    #     set_env_var('EA_INIT_SIGMA','0.1')
    #     set_env_var('EA_STEP_SIGMA','0.1')
    #     set_env_var('EA_STEP_MU','0.2')
    #     set_env_var('EA_MOMENTUM','0.8')
    #     set_env_var('SIM_TYPE', 'walls')
    #     set_env_var('N','1')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN18_FNN2x64_p50e20_vis32_fov97_rep{x}')
    #     start_EA()


if __name__ == '__main__':

    EA_runner()