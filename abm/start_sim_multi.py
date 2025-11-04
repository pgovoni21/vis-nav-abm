from contextlib import ExitStack
from pathlib import Path
import dotenv as de
import os
import numpy as np

from abm.NN.model import WorldModel as Model


def start(model_tuple=None, pvs=None, load_dir=None, seed=None, env_path=None, init_info=None): # "abm-start" in terminal

    # print(f'Running {save_ext}')

    if pvs is None: # if called from abm-start
        envconf = de.dotenv_values(Path(__file__).parent.parent / '.env')

        N = int(envconf["N"])
        models = []
        for i in range(N):
            NN, arch = reconstruct_NN(envconf)
            models.append(NN)

        envconf['WITH_VISUALIZATION'] = 0

    else:
        if env_path is None: # if called from EA
            envconf = de.dotenv_values(load_dir / '.env')
            arch, activ, RNN_type = model_tuple
            models = []
            for pv in pvs:
                NN = Model(arch, activ, RNN_type, pv)
                models.append(NN)

            envconf['WITH_VISUALIZATION'] = 0
            envconf['PLOT_TRAJECTORY'] = 0
        
        else: # if called from pickled NN
            envconf = de.dotenv_values(env_path)

            # envconf['WITH_VISUALIZATION'] = 1
            # envconf['INIT_FRAMERATE'] = 50
            envconf['N'] = len(pvs)

            # envconf['SOCIAL_INIT_TYPE'] = 'area'
            # envconf['SOCIAL_INIT_RANGE'] = 100

            models = []
            for pv in pvs:
                NN, arch = reconstruct_NN(envconf, pv)
                models.append(NN)

            ## social_rand perturbs
            # envconf['N'] = 1 # nosocial
            # envconf['SIM_TYPE'] = 'walls, social-ghostexploiter' # ghostexploiter
            # envconf['SIM_TYPE'] = 'walls, social-ghostexplorer' # ghostexplorer
            # envconf['SIM_TYPE'] = 'walls, social' # normal (perturb for the trained-perturb runs)


    # to run headless
    if int(envconf['WITH_VISUALIZATION']) == 0:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    # Set seed according to EA parent function to circumvent multiprocessing bug
    np.random.seed(seed)

    # import sim type
    from abm.simulation.sims_target_social_multievo import Simulation
    with ExitStack():
        sim = Simulation(env_size              =tuple(eval(envconf["ENV_SIZE"])),
                        window_pad             =int(envconf["WINDOW_PAD"]),
                        N                      =int(envconf["N"]),
                        T                      =int(envconf["T"]),
                        with_visualization     =bool(int(envconf["WITH_VISUALIZATION"])),
                        framerate              =int(envconf["INIT_FRAMERATE"]),
                        print_enabled          =bool(int(envconf["PRINT_ENABLED"])),
                        plot_trajectory        =bool(int(envconf["PLOT_TRAJECTORY"])),
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
                        NNs                    =models,
                        other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
                        vis_transform          =str(envconf["VIS_TRANSFORM"]),
                        percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
                        percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
                        action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
                        boundary_scale         =int(envconf["BOUNDARY_SCALE"]),
                        sim_type               =str(envconf["SIM_TYPE"]),
                        RW_rot_diff            =float(envconf["RW_ROT_DIFF"]),
                        social_init_range      =float(envconf["SOCIAL_INIT_RANGE"]),
                        social_init_type       =str(envconf["SOCIAL_INIT_TYPE"]),
                        init_info              =init_info
                        )
        t, dist, elapsed_time, data_agent = sim.start()
        # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {t, dist}')
    return t, dist, data_agent



def reconstruct_NN(envconf,pv=None):
    """mirrors start_EA arch packaging"""
    
    # gather NN variables
    num_class_elements   = 6 # multi-agent --> perception of 4 walls + 2 agent modes
    
    # assemble NN architecture
    vis_field_res        = int(envconf["VISUAL_FIELD_RESOLUTION"])
    CNN_input_size       = (num_class_elements, vis_field_res)
    CNN_depths           = list(map(int,envconf["CNN_DEPTHS"].split(',')))
    CNN_dims             = list(map(int,envconf["CNN_DIMS"].split(',')))
    RNN_other_input_size = int(envconf["RNN_OTHER_INPUT_SIZE"])
    RNN_hidden_size      = int(envconf["RNN_HIDDEN_SIZE"])
    LCL_output_size      = int(envconf["LCL_OUTPUT_SIZE"])
    misc_weight          = float(envconf["MISC_WEIGHT"])

    arch = (
        CNN_input_size, 
        CNN_depths, 
        CNN_dims, 
        RNN_other_input_size, 
        RNN_hidden_size, 
        LCL_output_size,
        misc_weight
        )

    activ                     =str(envconf["NN_ACTIVATION_FUNCTION"])
    RNN_type                  =str(envconf["RNN_TYPE"])

    NN = Model(arch, activ, RNN_type, pv)

    return NN, arch


if __name__ == '__main__':

    import pickle
    from abm.monitoring.trajs import find_top_val_gen

    # load param_vec + env_path
    data_dir = Path(__file__).parent / r'data/simulation_data/'

    names = []
    names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep9') # vids - indiv nav
    names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep4') # vids - follower
    names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep4') # vids - follower
    names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep4') # vids - follower
    names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep4') # vids - follower
    # names.append('sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep12') # vids - follower
    # names.append('sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep12') # vids - follower
    # names.append('sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep12') # vids - follower
    # names.append('sc_N4_NRW1_ND2_CNN14_FNN16_vis8_rep12') # vids - follower
    # names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep9') # vids - indiv nav
    # names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep9') # vids - indiv nav
    # names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep9') # vids - indiv nav
    # names.append('sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep9') # vids - indiv nav

    pvs = []
    for exp_name in names:
        gen_ext, valfit = find_top_val_gen(exp_name, 'cen')
        # NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
        NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
        env_path = fr'{data_dir}/{exp_name}/.env'

        with open(NN_pv_path,'rb') as f:
            pv = pickle.load(f)
        pvs.append(pv)
    

    start(pvs=pvs, env_path=env_path, seed=1)
    # start()
