from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    # # gadus
    # for x in range(18):
    #     set_env_var('EA_START_SEED',40000)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_seed40k_rep{x+2}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('EA_START_SEED',1000)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'silu')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_silu_rep{x}')
    #     start_EA()

    # # # fish
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (500,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_res50_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_res40_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (300,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_res30_rep{x}')
    #     start_EA()
    # for x in range(17):
    #     set_env_var('RESOURCE_POS', (200,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_res20_rep{x+3}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (100,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_res10_rep{x}')
    #     start_EA()


    # # # compute4
    # for x in range(1):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 8)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 0)
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x+19}')
    #     start_EA()
    # for x in range(3):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 8)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', .1)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 0)
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_angl_n10_rep{x+17}')
    #     start_EA()
    # for x in range(15):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 12)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 0)
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x+5}')
    #     start_EA()

    # # # compute2
    # for x in range(20):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 10)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 0)
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_rep{x}')
    #     start_EA()

    # # # compute 12
    # for x in range(20):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 12)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 100)
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmn100_rep{x}')
    #     start_EA()

    # # # compute 13
    # for x in range(20):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 10)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', .1)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 0)
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_angl_n10_rep{x}')
    #     start_EA()


    # # # michaelis
    # for x in range(20):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 12)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', .1)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 0)
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_angl_n10_rep{x}')
    #     start_EA()
    # for x in range(10):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 8)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 0)
    #     set_env_var('RNN_TYPE', 'fnn_noise')
    #     set_env_var('RNN_HIDDEN_SIZE', 8)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNNn8_p50e20_vis8_lm100_rep{x+10}')
    #     start_EA()

    # # # menten
    # for x in range(20):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 10)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 0)
    #     set_env_var('RNN_TYPE', 'fnn_noise')
    #     set_env_var('RNN_HIDDEN_SIZE', 8)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNNn8_p50e20_vis10_lm100_rep{x}')
    #     start_EA()
    # for x in range(10):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 8)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('PERCEP_LM_NOISE_STD', 0)
    #     set_env_var('RNN_TYPE', 'fnn_noise')
    #     set_env_var('RNN_HIDDEN_SIZE', 8)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNNn8_p50e20_vis8_lm100_rep{x}')
    #     start_EA()


    
    # # # compute 13

    # # # michaelis

    # # # menten

    # # # compute 10

    # # # compute 8

    # # # compute 7

    # # # weierstrass


    # for x in range(20):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'silu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 8)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_silu_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 8)
    #     set_env_var('AGENT_FOV', .7)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_fov7_rep{x}')
    #     start_EA()

if __name__ == '__main__':

    EA_runner()