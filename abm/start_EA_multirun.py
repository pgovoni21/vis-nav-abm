from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    # # # gadus
    for x in range(20):
        set_env_var('EA_START_SEED', '10000')
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('EA_START_SEED', '20000')
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed20k_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('EA_START_SEED', '30000')
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed30k_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('EA_START_SEED', '40000')
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed40k_rep{x}')
        start_EA()

    # # # fish
    for x in range(20):
        set_env_var('RESOURCE_POS', (0,0))
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res0_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('RESOURCE_POS', (200,0))
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res20_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('RESOURCE_POS', (500,0))
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res50_rep{x}')
        start_EA()


    # # # compute 2
    for x in range(20):
        set_env_var('RESOURCE_POS', (500,500))
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res55_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('RESOURCE_POS', (500,300))
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res53_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('RESOURCE_POS', (500,100))
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res51_rep{x}')
        start_EA()

    # # # menten
    for x in range(20):
        set_env_var('RESOURCE_POS', (300,100))
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res31_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('RESOURCE_POS', (400,200))
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res42_rep{x}')
        start_EA()
    for x in range(20):
        set_env_var('RESOURCE_POS', (200,200))
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res22_rep{x}')
        start_EA()
 





    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (500,400))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res54_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (500,200))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res52_rep{x}')
    #     start_EA()
    

    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,300))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res43_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,100))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res41_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (400,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res40_rep{x}')
    #     start_EA()
    
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (300,300))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res33_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (300,200))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res32_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (300,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res30_rep{x}')
    #     start_EA()
    
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (200,100))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res21_rep{x}')
    #     start_EA()
    
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (100,100))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res11_rep{x}')
    #     start_EA()
    # for x in range(20):
    #     set_env_var('RESOURCE_POS', (100,0))
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_res10_rep{x}')
    #     start_EA()


if __name__ == '__main__':

    EA_runner()