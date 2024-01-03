from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    # # # gadus
    # for x in range(20):
    #     set_env_var('EA_START_SEED', '10000')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('EA_START_SEED', '20000')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed20k_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('EA_START_SEED', '30000')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed30k_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('EA_START_SEED', '40000')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed40k_rep{x}')
    #     start_EA()

    # # # fish
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (0,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res0_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (200,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res20_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (500,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res50_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res40_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (300,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res30_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (100,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res10_rep{x}')
    #     start_EA()


    # # # compute 2
    # for x in range(16):
    #     set_env_var('RESOURCE_POS', (500,400))
    #     set_env_var('PERCEP_ANGLE_NOISE', 0)
    #     set_env_var('ACTION_NOISE', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res54_rep{x+4}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (500,200))
    #     set_env_var('PERCEP_ANGLE_NOISE', 0)
    #     set_env_var('ACTION_NOISE', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res52_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE', 0)
    #     set_env_var('ACTION_NOISE', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 8)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN8_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE', 0)
    #     set_env_var('ACTION_NOISE', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 16)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    #     start_EA()

    
    # # # compute 4
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (200,100))
    #     set_env_var('PERCEP_ANGLE_NOISE', 0)
    #     set_env_var('ACTION_NOISE', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res21_rep{x}')
    #     start_EA()
    # for x in range(16):
    #     set_env_var('RESOURCE_POS', (300,300))
    #     set_env_var('PERCEP_ANGLE_NOISE', 0)
    #     set_env_var('ACTION_NOISE', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res33_rep{x+4}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE', 0)
    #     set_env_var('ACTION_NOISE', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 3)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN3_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE', 0)
    #     set_env_var('ACTION_NOISE', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 4)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN4_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    #     start_EA()


    # # # michaelis
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', .1)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n10_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', .2)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n20_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', .05)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n05_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', .15)
    #     set_env_var('ACTION_NOISE_STD', 0)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n15_rep{x}')
    #     start_EA()

    # # # menten
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', .1)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n10_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', .15)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n15_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', .05)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n05_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,400))
    #     set_env_var('PERCEP_ANGLE_NOISE_STD', 0)
    #     set_env_var('ACTION_NOISE_STD', .2)
    #     set_env_var('RNN_HIDDEN_SIZE', 2)
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_act_n20_rep{x}')
    #     start_EA()



if __name__ == '__main__':

    EA_runner()