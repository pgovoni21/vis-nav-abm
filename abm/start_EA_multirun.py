from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    # # gadus
    for x in range(20):
        set_env_var('RADIUS_LANDMARK', 100)
        set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
        set_env_var('VISUAL_FIELD_RESOLUTION', 12)
        set_env_var('AGENT_FOV', .4)
        set_env_var('PERCEP_ANGLE_NOISE_STD', .05)
        set_env_var('ACTION_NOISE_STD', 0)
        set_env_var('LM_DIST_NOISE_STD', 50) 
        set_env_var('LM_ANGLE_NOISE_STD', .05) 
        set_env_var('LM_RADIUS_NOISE_STD', 50) 
        set_env_var('RNN_TYPE', 'fnn')
        set_env_var('RNN_HIDDEN_SIZE', 2)
        set_env_var('VIS_TRANSFORM', '') 
        set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_anglselflm_05_lmdistrad_n050_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('RADIUS_LANDMARK', 100)
        set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
        set_env_var('VISUAL_FIELD_RESOLUTION', 16)
        set_env_var('AGENT_FOV', .4)
        set_env_var('PERCEP_ANGLE_NOISE_STD', .05)
        set_env_var('ACTION_NOISE_STD', 0)
        set_env_var('LM_DIST_NOISE_STD', 50) 
        set_env_var('LM_ANGLE_NOISE_STD', .05) 
        set_env_var('LM_RADIUS_NOISE_STD', 50) 
        set_env_var('RNN_TYPE', 'fnn')
        set_env_var('RNN_HIDDEN_SIZE', 2)
        set_env_var('VIS_TRANSFORM', '') 
        set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_anglselflm_05_lmdistrad_n050_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('RADIUS_LANDMARK', 100)
        set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
        set_env_var('VISUAL_FIELD_RESOLUTION', 10)
        set_env_var('AGENT_FOV', .4)
        set_env_var('PERCEP_ANGLE_NOISE_STD', .05)
        set_env_var('ACTION_NOISE_STD', 0)
        set_env_var('LM_DIST_NOISE_STD', 50) 
        set_env_var('LM_ANGLE_NOISE_STD', .05) 
        set_env_var('LM_RADIUS_NOISE_STD', 50) 
        set_env_var('RNN_TYPE', 'fnn')
        set_env_var('RNN_HIDDEN_SIZE', 2)
        set_env_var('VIS_TRANSFORM', '') 
        set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_anglselflm_05_lmdistrad_n050_rep{x}')
        start_EA()
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



    # set_env_var('RADIUS_LANDMARK', 100)
    # set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    # set_env_var('VISUAL_FIELD_RESOLUTION', 12)
    # set_env_var('AGENT_FOV', .4)
    # set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    # set_env_var('ACTION_NOISE_STD', 0)
    # set_env_var('LM_DIST_NOISE_STD', 50) 
    # set_env_var('LM_ANGLE_NOISE_STD', 0) 
    # set_env_var('LM_RADIUS_NOISE_STD', 0) 
    # set_env_var('RNN_TYPE', 'fnn')
    # set_env_var('RNN_HIDDEN_SIZE', 2)
    # set_env_var('VIS_TRANSFORM', '') 
    # set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdist_n050_rep9')
    # start_EA()
    # for x in range(10):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 12)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('LM_DIST_NOISE_STD', 0) 
    #     set_env_var('LM_ANGLE_NOISE_STD', .1) 
    #     set_env_var('LM_RADIUS_NOISE_STD', 0) 
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('VIS_TRANSFORM', '') 
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10_rep{x}')
    #     start_EA()
    # for x in range(10):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 12)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('LM_DIST_NOISE_STD', 0) 
    #     set_env_var('LM_ANGLE_NOISE_STD', .1) 
    #     set_env_var('LM_RADIUS_NOISE_STD', 0) 
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('VIS_TRANSFORM', '') 
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10_rep{x+10}')
    #     start_EA()




    # for x in range(10):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 12)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('LM_DIST_NOISE_STD', 0) 
    #     set_env_var('LM_ANGLE_NOISE_STD', 0) 
    #     set_env_var('LM_RADIUS_NOISE_STD', 100) 
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('VIS_TRANSFORM', '') 
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n100_rep{x}')
    #     start_EA()
    # for x in range(10):
    #     set_env_var('RADIUS_LANDMARK', 100)
    #     set_env_var('NN_ACTIVATION_FUNCTION', 'relu')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', 12)
    #     set_env_var('AGENT_FOV', .4)
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('LM_DIST_NOISE_STD', 0) 
    #     set_env_var('LM_ANGLE_NOISE_STD', 0) 
    #     set_env_var('LM_RADIUS_NOISE_STD', 100) 
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('VIS_TRANSFORM', '') 
    #     set_env_var('EA_SAVE_NAME', f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n100_rep{x+10}')
    #     start_EA()


if __name__ == '__main__':

    EA_runner()