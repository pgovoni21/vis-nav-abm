# from abm.simulation.sims import Simulation
# from abm.simulation.sims_target import Simulation
# from abm.simulation.sims_target_double import Simulation
# from abm.simulation.sims_target_cross import Simulation
from abm.simulation.sims_target_LM import Simulation

from contextlib import ExitStack
from pathlib import Path
import dotenv as de
import os
import numpy as np

from abm.NN.model import WorldModel as Model


def start(model_tuple=None, pv=None, load_dir=None, seed=None, env_path=None): # "abm-start" in terminal

    # print(f'Running {save_ext}')

    if pv is None: # if called from abm-start
        envconf = de.dotenv_values(Path(__file__).parent.parent / '.env')
        NN = Model()

        envconf['WITH_VISUALIZATION'] = 1

    else:
        if env_path is None: # if called from EA
            envconf = de.dotenv_values(load_dir / '.env')
            arch, activ, RNN_type = model_tuple
            NN = Model(arch, activ, RNN_type, pv)

            envconf['WITH_VISUALIZATION'] = 0
            envconf['PLOT_TRAJECTORY'] = 0
        
        else: # if called from pickled NN
            envconf = de.dotenv_values(env_path)

            # override original EA-written env dict
            # envconf['LOG_ZARR_FILE'] = 0

            # envconf['WITH_VISUALIZATION'] = 1
            # envconf['INIT_FRAMERATE'] = 40

            # envconf['N'] = 4
            # envconf['T'] = 100000
            # envconf['RADIUS_RESOURCE'] = 100
            # envconf['MAXIMUM_VELOCITY'] = 5

            # envconf['VIS_TRANSFORM'] = ''
            # envconf['PERCEP_DIST_NOISE_STD'] = 0
            # envconf['PERCEP_ANGLE_NOISE_STD'] = .2
            # envconf['ACTION_NOISE_STD'] = .1
            # print(envconf['ENV_SIZE'])
            # print(envconf['RESOURCE_POS'])
            # print(envconf['VIS_TRANSFORM'])
            # print(envconf['SENSORY_NOISE_STD'])
            # print(envconf['PERCEP_ANGLE_NOISE_STD'])
            # print(envconf['ACTION_NOISE_STD'])
            # print(envconf['PERCEP_DIST_NOISE_STD'])

            NN, arch = reconstruct_NN(envconf, pv)

    # to run headless
    if int(envconf['WITH_VISUALIZATION']) == 0:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    # Set seed according to EA parent function to circumvent multiprocessing bug
    np.random.seed(seed)

    with ExitStack():
        sim = Simulation(env_size               =tuple(eval(envconf["ENV_SIZE"])),
                         window_pad             =int(envconf["WINDOW_PAD"]),
                         N                      =int(envconf["N"]),
                         T                      =int(envconf["T"]),
                         with_visualization     =bool(int(envconf["WITH_VISUALIZATION"])),
                         framerate              =int(envconf["INIT_FRAMERATE"]),
                         print_enabled          =bool(int(envconf["PRINT_ENABLED"])),
                         plot_trajectory        =bool(int(envconf["PLOT_TRAJECTORY"])),
                         log_zarr_file          =bool(int(envconf["LOG_ZARR_FILE"])),
                         save_ext               =None,
                         agent_radius           =int(envconf["RADIUS_AGENT"]),
                         max_vel                =int(envconf["MAXIMUM_VELOCITY"]),
                         vis_field_res          =int(envconf["VISUAL_FIELD_RESOLUTION"]),
                         vision_range           =int(envconf["VISION_RANGE"]),
                         agent_fov              =float(envconf['AGENT_FOV']),
                         show_vision_range      =bool(int(envconf["SHOW_VISION_RANGE"])),
                         agent_consumption      =int(envconf["AGENT_CONSUMPTION"]),
                         N_res                  =int(envconf["N_RESOURCES"]),
                         patch_radius           =float(envconf["RADIUS_RESOURCE"]),
                         res_pos                =tuple(eval(envconf["RESOURCE_POS"])),
                         res_units              =tuple(eval(envconf["RESOURCE_UNITS"])),
                         res_quality            =tuple(eval(envconf["RESOURCE_QUALITY"])),
                         regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
                         landmark_radius        =int(envconf["RADIUS_LANDMARK"]),
                         NN                     =NN,
                         other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
                         vis_transform          =str(envconf["VIS_TRANSFORM"]),
                         percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
                         percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
                         action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
                         )
        t, d, elapsed_time = sim.start()

        # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {t, d}')

    return t, d

def reconstruct_NN(envconf,pv):
    """mirrors start_EA arch packaging"""
    
    # gather NN variables
    N                    = int(envconf["N"])

    if N == 1:  num_class_elements = 4 # single-agent --> perception of 4 walls
    else:       num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes
    # num_class_elements = 4
    
    # assemble NN architecture
    vis_field_res        = int(envconf["VISUAL_FIELD_RESOLUTION"])
    CNN_input_size       = (num_class_elements, vis_field_res)
    CNN_depths           = list(map(int,envconf["CNN_DEPTHS"].split(',')))
    CNN_dims             = list(map(int,envconf["CNN_DIMS"].split(',')))
    RNN_other_input_size = int(envconf["RNN_OTHER_INPUT_SIZE"])
    RNN_hidden_size      = int(envconf["RNN_HIDDEN_SIZE"])
    LCL_output_size      = int(envconf["LCL_OUTPUT_SIZE"])

    arch = (
        CNN_input_size, 
        CNN_depths, 
        CNN_dims, 
        RNN_other_input_size, 
        RNN_hidden_size, 
        LCL_output_size,
        )

    activ                     =str(envconf["NN_ACTIVATION_FUNCTION"])
    RNN_type                  =str(envconf["RNN_TYPE"])

    NN = Model(arch, activ, RNN_type, pv)

    return NN, arch


if __name__ == '__main__':

    import pickle

    # load param_vec + env_path
    data_dir = Path(__file__).parent / r'data/simulation_data/'



    # sims = [f'sc_CNN1124_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(53)]
    # for exp_name in sims:
    #     # print(exp_name)
    #     env_path = fr'{data_dir}/{exp_name}/.env'
    #     envconf = de.dotenv_values(env_path)
    #     # print(envconf['RNN_TYPE'])
    #     # print(envconf['RNN_HIDDEN_SIZE'])
    #     # print(envconf['CNN_DEPTHS'])
    #     # print(envconf['CNN_DIMS'])
    #     # print(envconf['VISUAL_FIELD_RESOLUTION'])
    #     # print(envconf['EA_MOMENTUM'])
    #     print(envconf['EA_STEP_MU'])


    # exp_name = 'sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18'
    # gen_ext = 'gen790'
    # exp_name = 'sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'
    # gen_ext = 'gen979'


    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'
    # gen_ext = 'gen956'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4'
    # gen_ext = 'gen941'

    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_minmax_rep0'
    # gen_ext = 'gen857'

    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep2'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep0'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep5'
    # gen_ext = 'gen963'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep3'
    # gen_ext = 'gen956'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep5'
    # gen_ext = 'gen937'

    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep2'
    # gen_ext = 'gen963'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n1_rep5'
    # gen_ext = 'gen900'

    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n10_rep6'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n10_rep9'
    # gen_ext = 'gen844'

    # exp_name = 'sc_lm_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_lm100_rep6'
    # gen_ext = 'gen963'
    # exp_name = 'sc_lm_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_lm300_rep1'
    # gen_ext = 'gen857'
    exp_name = 'sc_lm_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_lm100_rep2'
    gen_ext = 'gen973'

    # NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
    env_path = fr'{data_dir}/{exp_name}/.env'

    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)

    start(pv=pv, env_path=env_path, seed=9)
