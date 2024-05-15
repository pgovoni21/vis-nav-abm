from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

    # for x in range(10):
    #     set_env_var('RNN_TYPE', 'fnn2')
    #     set_env_var('RNN_HIDDEN_SIZE', '64')
    #     set_env_var('LCL_OUTPUT_SIZE', '32')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN64x64_p50e20_vis8_PGPE_ss20_mom8_act32_rep{x}')
    #     start_EA()
    # for x in range(10):
    #     set_env_var('RNN_TYPE', 'fnn2')
    #     set_env_var('RNN_HIDDEN_SIZE', '64')
    #     set_env_var('LCL_OUTPUT_SIZE', '32')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN64x64_p50e20_vis8_PGPE_ss20_mom8_act32_rep{x+10}')
    #     start_EA()
    

    # for x in range(10):
    #     set_env_var('RNN_TYPE', 'fnn2')
    #     set_env_var('RNN_HIDDEN_SIZE', '16')
    #     set_env_var('LCL_OUTPUT_SIZE', '8')
    #     set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN16x16_p50e20_vis8_PGPE_ss20_mom8_act8_rep{x}')
    #     start_EA()
    for x in range(10):
        set_env_var('RNN_TYPE', 'fnn2')
        set_env_var('RNN_HIDDEN_SIZE', '64')
        set_env_var('LCL_OUTPUT_SIZE', '8')
        set_env_var('EA_SAVE_NAME', f'sc_CNN14_FNN16x16_p50e20_vis8_PGPE_ss20_mom8_act8_rep{x+10}')
        start_EA()



if __name__ == '__main__':

    EA_runner()