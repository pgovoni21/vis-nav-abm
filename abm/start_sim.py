# from abm.simulation.sims import Simulation
# from abm.simulation.sims_target import Simulation
from abm.simulation.sims_target_double import Simulation
# from abm.simulation.sims_target_cross import Simulation

from contextlib import ExitStack
from pathlib import Path
import dotenv as de
import os
import numpy as np

from abm.NN.model import WorldModel as Model
# from abm.NN.model_simp import WorldModel as Model


def start(model_tuple=None, pv=None, save_ext=None, seed=None, env_path=None): # "abm-start" in terminal

    # print(f'Running {save_ext}')

    if pv is None: # if called from abm-start
        envconf = de.dotenv_values(Path(__file__).parent.parent / '.env')
        NN = None
    else:
        if env_path is None: # if called from EA
            envconf = de.dotenv_values(Path(__file__).parent.parent / '.env')
            arch, activ, RNN_type = model_tuple
            NN = Model(arch, activ, RNN_type, pv)

            envconf['WITH_VISUALIZATION'] = 0
            envconf['PLOT_TRAJECTORY'] = 0
        
        else: # if called directly with pickled NN
            envconf = de.dotenv_values(env_path)
            NN, arch = reconstruct_NN(envconf, pv)

            # override original EA-written env dict
            # envconf['WITH_VISUALIZATION'] = 1
            # envconf['PLOT_TRAJECTORY'] = 0
            # envconf['LOG_ZARR_FILE'] = 0
            # envconf['INIT_FRAMERATE'] = 50
            envconf['WITH_VISUALIZATION'] = 0
            envconf['PLOT_TRAJECTORY'] = 1
            envconf['LOG_ZARR_FILE'] = 1

    # to run headless
    if int(envconf['WITH_VISUALIZATION']) == 0:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    # Set seed according to EA parent function to circumvent multiprocessing bug
    np.random.seed(seed)

    with ExitStack():
        sim = Simulation(width                  =int(envconf["ENV_WIDTH"]),
                         height                 =int(envconf["ENV_HEIGHT"]),
                         window_pad             =int(envconf["WINDOW_PAD"]),
                         N                      =int(envconf["N"]),
                         T                      =int(envconf["T"]),
                         with_visualization     =bool(int(envconf["WITH_VISUALIZATION"])),
                         framerate              =int(envconf["INIT_FRAMERATE"]),
                         print_enabled          =bool(int(envconf["PRINT_ENABLED"])),
                         plot_trajectory        =bool(int(envconf["PLOT_TRAJECTORY"])),
                         log_zarr_file          =bool(int(envconf["LOG_ZARR_FILE"])),
                         save_ext               =save_ext,
                         agent_radius           =int(envconf["RADIUS_AGENT"]),
                         max_vel                =int(envconf["MAXIMUM_VELOCITY"]),
                         collision_slowdown     =float(envconf["COLLISION_SLOWDOWN"]),
                         vis_field_res          =int(envconf["VISUAL_FIELD_RESOLUTION"]),
                         contact_field_res      =int(envconf["CONTACT_FIELD_RESOLUTION"]),
                         collide_agents         =bool(int(envconf["AGENT_AGENT_COLLISION"])),
                         vision_range           =int(envconf["VISION_RANGE"]),
                         agent_fov              =float(envconf['AGENT_FOV']),
                         visual_exclusion       =bool(int(envconf["VISUAL_EXCLUSION"])),
                         show_vision_range      =bool(int(envconf["SHOW_VISION_RANGE"])),
                         agent_consumption      =int(envconf["AGENT_CONSUMPTION"]),
                         N_resrc                =int(envconf["N_RESOURCES"]),
                         patch_radius           =float(envconf["RADIUS_RESOURCE"]),
                         min_resrc_perpatch     =int(envconf["MIN_RESOURCE_PER_PATCH"]),
                         max_resrc_perpatch     =int(envconf["MAX_RESOURCE_PER_PATCH"]),
                         min_resrc_quality      =float(envconf["MIN_RESOURCE_QUALITY"]),
                         max_resrc_quality      =float(envconf["MAX_RESOURCE_QUALITY"]),
                         regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
                         NN                     =NN,
                         CNN_depths             =list(map(int,envconf["CNN_DEPTHS"].split(','))),
                         CNN_dims               =list(map(int,envconf["CNN_DIMS"].split(','))),
                         RNN_input_other_size   =int(envconf["RNN_INPUT_OTHER_SIZE"]),
                         RNN_hidden_size        =int(envconf["RNN_HIDDEN_SIZE"]),
                         LCL_output_size        =int(envconf["LCL_OUTPUT_SIZE"]),
                         NN_activ               =str(envconf["NN_ACTIVATION_FUNCTION"]),
                         RNN_type               =str(envconf["RNN_TYPE"]),
                         )
        fitnesses, elapsed_time = sim.start()

        # print(f'Finished {save_ext}, runtime: {elapsed_time} sec, fitness: {fitnesses[0]}')

    return save_ext, fitnesses, elapsed_time


def reconstruct_NN(envconf,pv):
    """mirrors start_EA arch packaging"""
    
    # gather NN variables
    N                    = int(envconf["N"])

    if N == 1:  num_class_elements = 4 # single-agent --> perception of 4 walls
    else:       num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes
    
    vis_field_res        = int(envconf["VISUAL_FIELD_RESOLUTION"])
    contact_field_res    = int(envconf["CONTACT_FIELD_RESOLUTION"])
    other_input_size     = int(envconf["RNN_INPUT_OTHER_SIZE"]) # last action + last velocity + on_resrc
    
    # assemble NN architecture
    CNN_input_size       = (num_class_elements, vis_field_res)
    CNN_depths           = list(map(int,envconf["CNN_DEPTHS"].split(',')))
    CNN_dims             = list(map(int,envconf["CNN_DIMS"].split(',')))
    RNN_other_input_size = (contact_field_res, other_input_size)
    RNN_hidden_size      = int(envconf["RNN_HIDDEN_SIZE"])
    LCL_output_size      = int(envconf["LCL_OUTPUT_SIZE"]) # dvel + dtheta

    arch = (
        CNN_input_size, 
        CNN_depths, 
        CNN_dims, 
        RNN_other_input_size, 
        RNN_hidden_size, 
        LCL_output_size
        )

    activ                     =str(envconf["NN_ACTIVATION_FUNCTION"])
    RNN_type                  =str(envconf["RNN_TYPE"])

    NN = Model(arch, activ, RNN_type, pv)

    return NN, arch


if __name__ == '__main__':

    import pickle

    # load param_vec + env_path
    data_dir = Path(__file__).parent / r'data/simulation_data/'


    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep0'
    # NN_ext = 'gen997/NN0_af7'
    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep1'
    # NN_ext = 'gen977/NN0_af6'
    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep2'
    # NN_ext = 'gen549/NN0_af8'
    # exp_name = 'doublepoint_CNN1128_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep3'
    # NN_ext = 'gen999/NN0_af7'
    exp_name = 'doublepoint_CNN1122_GRU2_p25e5g1000_sig0p1_vis8_dirfit_rep0'
    NN_ext = 'gen969/NN0_af7'

    # exp_name = 'test'
    # NN_ext = 'gen0_NN0'
    # NN_pv_path = fr'{data_dir}/{exp_name}/{NN_ext}.bin'


    NN_pv_path = fr'{data_dir}/{exp_name}/{NN_ext}/NN_pickle.bin'
    print(NN_pv_path)
    env_path = fr'{data_dir}/{exp_name}/.env'

    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)

    start(pv=pv, env_path=env_path, seed=6)

    # for s in [0,1,2]:
    # # for s in [0]:
    #     start(pv=pv, env_path=env_path, save_ext=f'1run_{exp_name}_seed{str(s)}', seed=s)