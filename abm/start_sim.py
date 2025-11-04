from contextlib import ExitStack
from pathlib import Path
import dotenv as de
import os
import numpy as np

from abm.NN.model import WorldModel as Model


def start(model_tuple=None, pv=None, load_dir=None, seed=None, env_path=None, init_info=None, feat_out=False): # "abm-start" in terminal

    # print(f'Running {save_ext}')

    if pv is None: # if called from abm-start
        envconf = de.dotenv_values(Path(__file__).parent.parent / '.env')
        NN, arch = reconstruct_NN(envconf)

        envconf['WITH_VISUALIZATION'] = 1
        # envconf['INIT_FRAMERATE'] = 10
        # envconf['AGENT_FOV'] = 1
        # envconf['VISUAL_FIELD_RESOLUTION'] = 32

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
            # envconf['INIT_FRAMERATE'] = 100

            # envconf['N'] = 1
            # envconf['T'] = 50000
            # envconf['RADIUS_RESOURCE'] = 100
            # envconf['MAXIMUM_VELOCITY'] = 5

            # envconf['SOCIAL_INIT_TYPE'] = 'ag'
            # envconf['SOCIAL_INIT_RANGE'] = 100

            # envconf['RNN_TYPE'] = 'fnn_random_as_choice'

            # envconf['SIM_TYPE'] = 'walls, markov'
            # envconf['BOUNDARY_SCALE'] = '0'
            # envconf['MISC_WEIGHT'] = 0.25
            # envconf['RESOURCE_UNITS'] = '(10000000,-1)'

            NN, arch = reconstruct_NN(envconf, pv)

            # feat_out = True

            ## social_rand perturbs
            # envconf['N'] = 1 # nosocial
            # envconf['SIM_TYPE'] = 'walls, social-ghostexploiter' # ghostexploiter
            # envconf['SIM_TYPE'] = 'walls, social-ghostexplorer' # ghostexplorer
            # envconf['SIM_TYPE'] = 'walls, social-ghostexplorers' # ghostexplorers
            # envconf['N'] = int(envconf['N']) + 2 # +2 direct agents
            # envconf['N_RAND'] = int(envconf['N_RAND']) + 2 # +2 random agents (needs above too)
            # envconf['SIM_TYPE'] = 'nowalls, social-ghostexploiter' # ghostexploiter in vacuum
            # envconf['SIM_TYPE'] = 'walls, social-model' # selfsocial

            # envconf['N_RAND'] = '5'

            # envconf['AGENT_COLLIDE'] = '1'

            # envconf['N'] = 2
            # envconf['SOCIAL_INIT_TYPE'] = 'ag' # init-ag-1
            # envconf['SOCIAL_INIT_RANGE'] = '100'

            # envconf['N'] = 6
            # envconf['SOCIAL_INIT_TYPE'] = 'res' # init-res-5
            # envconf['SOCIAL_INIT_RANGE'] = '100'

    # to run headless
    if int(envconf['WITH_VISUALIZATION']) == 0:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    # Set seed according to EA parent function to circumvent multiprocessing bug
    np.random.seed(seed)

    # import sim type
    if envconf['SIM_TYPE'] == 'walls' or envconf['SIM_TYPE'] == 'walls, 2x pinball':
        from abm.simulation.sims_target import Simulation
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
                            NN                     =NN,
                            other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
                            vis_transform          =str(envconf["VIS_TRANSFORM"]),
                            percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
                            percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
                            action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
                            boundary_scale         =int(envconf["BOUNDARY_SCALE"]),
                            sim_type               =str(envconf["SIM_TYPE"]),
                            )
            t, dist, elapsed_time = sim.start()
            # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {t, dist}')
        return t, dist, None

    elif 'social' in envconf['SIM_TYPE']:
        from abm.simulation.sims_target_social import Simulation
        with ExitStack():
            sim = Simulation(env_size              =tuple(eval(envconf["ENV_SIZE"])),
                            window_pad             =int(envconf["WINDOW_PAD"]),
                            N                      =int(envconf["N"]),
                            N_rand                 =int(envconf["N_RAND"]),
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
                            agent_collide          =bool(int(envconf["AGENT_COLLIDE"])),
                            agent_patch_collide    =bool(int(envconf["AGENT_PATCH_COLLIDE"])),
                            N_res                  =int(envconf["N_RESOURCES"]),
                            patch_radius           =float(envconf["RADIUS_RESOURCE"]),
                            res_pos                =tuple(eval(envconf["RESOURCE_POS"])),
                            res_units              =tuple(eval(envconf["RESOURCE_UNITS"])),
                            res_quality            =tuple(eval(envconf["RESOURCE_QUALITY"])),
                            regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
                            NN                     =NN,
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
                            init_info              =init_info,
                            feat_out               =feat_out,
                            )
            t, dist, elapsed_time, data_agent = sim.start()
            # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {t, int(dist)}')
        return t, dist, data_agent

    # elif envconf['SIM_TYPE'] == 'walls, social-test':
    #     from abm.simulation.sims_target_social_rand_test import Simulation
    #     with ExitStack():
    #         sim = Simulation(env_size              =tuple(eval(envconf["ENV_SIZE"])),
    #                         window_pad             =int(envconf["WINDOW_PAD"]),
    #                         N                      =int(envconf["N"]),
    #                         N_rand                 =int(envconf["N_RAND"]),
    #                         T                      =int(envconf["T"]),
    #                         with_visualization     =bool(int(envconf["WITH_VISUALIZATION"])),
    #                         framerate              =int(envconf["INIT_FRAMERATE"]),
    #                         print_enabled          =bool(int(envconf["PRINT_ENABLED"])),
    #                         plot_trajectory        =bool(int(envconf["PLOT_TRAJECTORY"])),
    #                         save_ext               =None,
    #                         agent_radius           =int(envconf["RADIUS_AGENT"]),
    #                         max_vel                =int(envconf["MAXIMUM_VELOCITY"]),
    #                         vis_field_res          =int(envconf["VISUAL_FIELD_RESOLUTION"]),
    #                         vision_range           =int(envconf["VISION_RANGE"]),
    #                         agent_fov              =float(envconf['AGENT_FOV']),
    #                         show_vision_range      =bool(int(envconf["SHOW_VISION_RANGE"])),
    #                         agent_consumption      =int(envconf["AGENT_CONSUMPTION"]),
    #                         N_res                  =int(envconf["N_RESOURCES"]),
    #                         patch_radius           =float(envconf["RADIUS_RESOURCE"]),
    #                         res_pos                =tuple(eval(envconf["RESOURCE_POS"])),
    #                         res_units              =tuple(eval(envconf["RESOURCE_UNITS"])),
    #                         res_quality            =tuple(eval(envconf["RESOURCE_QUALITY"])),
    #                         regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
    #                         NN                     =NN,
    #                         other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
    #                         vis_transform          =str(envconf["VIS_TRANSFORM"]),
    #                         percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
    #                         percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
    #                         action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
    #                         boundary_scale         =int(envconf["BOUNDARY_SCALE"]),
    #                         sim_type               =str(envconf["SIM_TYPE"]),
    #                         RW_rot_diff            =float(envconf["RW_ROT_DIFF"]),
    #                         )
    #         t, dist, elapsed_time = sim.start()
    #         # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {t, dist}')
    #     return t, dist

    # elif envconf['SIM_TYPE'] == 'walls, social-model':
    #     from abm.simulation.sims_target_social_model import Simulation
    #     with ExitStack():
    #         sim = Simulation(env_size              =tuple(eval(envconf["ENV_SIZE"])),
    #                         window_pad             =int(envconf["WINDOW_PAD"]),
    #                         N                      =int(envconf["N"]),
    #                         T                      =int(envconf["T"]),
    #                         with_visualization     =bool(int(envconf["WITH_VISUALIZATION"])),
    #                         framerate              =int(envconf["INIT_FRAMERATE"]),
    #                         print_enabled          =bool(int(envconf["PRINT_ENABLED"])),
    #                         plot_trajectory        =bool(int(envconf["PLOT_TRAJECTORY"])),
    #                         save_ext               =None,
    #                         agent_radius           =int(envconf["RADIUS_AGENT"]),
    #                         max_vel                =int(envconf["MAXIMUM_VELOCITY"]),
    #                         vis_field_res          =int(envconf["VISUAL_FIELD_RESOLUTION"]),
    #                         vision_range           =int(envconf["VISION_RANGE"]),
    #                         agent_fov              =float(envconf['AGENT_FOV']),
    #                         show_vision_range      =bool(int(envconf["SHOW_VISION_RANGE"])),
    #                         agent_consumption      =int(envconf["AGENT_CONSUMPTION"]),
    #                         N_res                  =int(envconf["N_RESOURCES"]),
    #                         patch_radius           =float(envconf["RADIUS_RESOURCE"]),
    #                         res_pos                =tuple(eval(envconf["RESOURCE_POS"])),
    #                         res_units              =tuple(eval(envconf["RESOURCE_UNITS"])),
    #                         res_quality            =tuple(eval(envconf["RESOURCE_QUALITY"])),
    #                         regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
    #                         NN                     =NN,
    #                         other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
    #                         vis_transform          =str(envconf["VIS_TRANSFORM"]),
    #                         percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
    #                         percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
    #                         action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
    #                         boundary_scale         =int(envconf["BOUNDARY_SCALE"]),
    #                         sim_type               =str(envconf["SIM_TYPE"]),
    #                         )
    #         t, dist, elapsed_time = sim.start()
    #         # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {t, dist}')
    #     return t, dist

    # elif envconf['SIM_TYPE'] == 'walls, markov':
    #     from abm.simulation.sims_target_markov import Simulation
    #     with ExitStack():
    #         sim = Simulation(env_size              =tuple(eval(envconf["ENV_SIZE"])),
    #                         window_pad             =int(envconf["WINDOW_PAD"]),
    #                         N                      =int(envconf["N"]),
    #                         T                      =int(envconf["T"]),
    #                         with_visualization     =bool(int(envconf["WITH_VISUALIZATION"])),
    #                         framerate              =int(envconf["INIT_FRAMERATE"]),
    #                         print_enabled          =bool(int(envconf["PRINT_ENABLED"])),
    #                         plot_trajectory        =bool(int(envconf["PLOT_TRAJECTORY"])),
    #                         log_zarr_file          =bool(int(envconf["LOG_ZARR_FILE"])),
    #                         save_ext               =None,
    #                         agent_radius           =int(envconf["RADIUS_AGENT"]),
    #                         max_vel                =int(envconf["MAXIMUM_VELOCITY"]),
    #                         vis_field_res          =int(envconf["VISUAL_FIELD_RESOLUTION"]),
    #                         vision_range           =int(envconf["VISION_RANGE"]),
    #                         agent_fov              =float(envconf['AGENT_FOV']),
    #                         show_vision_range      =bool(int(envconf["SHOW_VISION_RANGE"])),
    #                         agent_consumption      =int(envconf["AGENT_CONSUMPTION"]),
    #                         N_res                  =int(envconf["N_RESOURCES"]),
    #                         patch_radius           =float(envconf["RADIUS_RESOURCE"]),
    #                         res_pos                =tuple(eval(envconf["RESOURCE_POS"])),
    #                         res_units              =tuple(eval(envconf["RESOURCE_UNITS"])),
    #                         res_quality            =tuple(eval(envconf["RESOURCE_QUALITY"])),
    #                         regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
    #                         NN                     =NN,
    #                         other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
    #                         vis_transform          =str(envconf["VIS_TRANSFORM"]),
    #                         percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
    #                         percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
    #                         action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
    #                         boundary_scale         =int(envconf["BOUNDARY_SCALE"]),
    #                         sim_type               =str(envconf["SIM_TYPE"]),
    #                         )
    #         t, dist, elapsed_time = sim.start()
    #         # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {t, dist}')
    #     return t, dist

    # elif envconf['SIM_TYPE'] == 'nowalls':
    #     from abm.simulation.sims_target_nowalls import Simulation
    #     with ExitStack():
    #         sim = Simulation(env_size              =tuple(eval(envconf["ENV_SIZE"])),
    #                         window_pad             =int(envconf["WINDOW_PAD"]),
    #                         N                      =int(envconf["N"]),
    #                         T                      =int(envconf["T"]),
    #                         with_visualization     =bool(int(envconf["WITH_VISUALIZATION"])),
    #                         framerate              =int(envconf["INIT_FRAMERATE"]),
    #                         print_enabled          =bool(int(envconf["PRINT_ENABLED"])),
    #                         plot_trajectory        =bool(int(envconf["PLOT_TRAJECTORY"])),
    #                         log_zarr_file          =bool(int(envconf["LOG_ZARR_FILE"])),
    #                         save_ext               =None,
    #                         agent_radius           =int(envconf["RADIUS_AGENT"]),
    #                         max_vel                =int(envconf["MAXIMUM_VELOCITY"]),
    #                         vis_field_res          =int(envconf["VISUAL_FIELD_RESOLUTION"]),
    #                         vision_range           =int(envconf["VISION_RANGE"]),
    #                         agent_fov              =float(envconf['AGENT_FOV']),
    #                         show_vision_range      =bool(int(envconf["SHOW_VISION_RANGE"])),
    #                         agent_consumption      =int(envconf["AGENT_CONSUMPTION"]),
    #                         N_res                  =int(envconf["N_RESOURCES"]),
    #                         patch_radius           =float(envconf["RADIUS_RESOURCE"]),
    #                         res_pos                =tuple(eval(envconf["RESOURCE_POS"])),
    #                         res_units              =tuple(eval(envconf["RESOURCE_UNITS"])),
    #                         res_quality            =tuple(eval(envconf["RESOURCE_QUALITY"])),
    #                         regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
    #                         NN                     =NN,
    #                         other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
    #                         vis_transform          =str(envconf["VIS_TRANSFORM"]),
    #                         percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
    #                         percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
    #                         action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
    #                         boundary_scale         =int(envconf["BOUNDARY_SCALE"]),
    #                         sim_type               =str(envconf["SIM_TYPE"]),
    #                         )
    #         t, res_collected, elapsed_time, first_consume_time = sim.start()
    #         # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {first_consume_time, res_collected}')
    #     return first_consume_time, res_collected


    # elif envconf['SIM_TYPE'].startswith('nowalls_ghost'):
    #     from abm.simulation.sims_target_nowalls_ghost import Simulation
    #     with ExitStack():
    #         sim = Simulation(env_size              =tuple(eval(envconf["ENV_SIZE"])),
    #                         window_pad             =int(envconf["WINDOW_PAD"]),
    #                         N                      =int(envconf["N"]),
    #                         T                      =int(envconf["T"]),
    #                         with_visualization     =bool(int(envconf["WITH_VISUALIZATION"])),
    #                         framerate              =int(envconf["INIT_FRAMERATE"]),
    #                         print_enabled          =bool(int(envconf["PRINT_ENABLED"])),
    #                         plot_trajectory        =bool(int(envconf["PLOT_TRAJECTORY"])),
    #                         log_zarr_file          =bool(int(envconf["LOG_ZARR_FILE"])),
    #                         save_ext               =None,
    #                         agent_radius           =int(envconf["RADIUS_AGENT"]),
    #                         max_vel                =int(envconf["MAXIMUM_VELOCITY"]),
    #                         vis_field_res          =int(envconf["VISUAL_FIELD_RESOLUTION"]),
    #                         vision_range           =int(envconf["VISION_RANGE"]),
    #                         agent_fov              =float(envconf['AGENT_FOV']),
    #                         show_vision_range      =bool(int(envconf["SHOW_VISION_RANGE"])),
    #                         agent_consumption      =int(envconf["AGENT_CONSUMPTION"]),
    #                         N_res                  =int(envconf["N_RESOURCES"]),
    #                         patch_radius           =float(envconf["RADIUS_RESOURCE"]),
    #                         res_pos                =tuple(eval(envconf["RESOURCE_POS"])),
    #                         res_units              =tuple(eval(envconf["RESOURCE_UNITS"])),
    #                         res_quality            =tuple(eval(envconf["RESOURCE_QUALITY"])),
    #                         regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
    #                         NN                     =NN,
    #                         other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
    #                         vis_transform          =str(envconf["VIS_TRANSFORM"]),
    #                         percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
    #                         percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
    #                         action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
    #                         boundary_scale         =int(envconf["BOUNDARY_SCALE"]),
    #                         sim_type               =str(envconf["SIM_TYPE"]),
    #                         )
    #         t, res_collected, elapsed_time, first_consume_time = sim.start()
    #         # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {first_consume_time, res_collected}')
    #     return first_consume_time, res_collected


    # elif envconf['SIM_TYPE'] == 'LM':
    #     from abm.simulation.sims_target_LM import Simulation
    #     with ExitStack():
    #         sim = Simulation(env_size               =tuple(eval(envconf["ENV_SIZE"])),
    #                         window_pad             =int(envconf["WINDOW_PAD"]),
    #                         N                      =int(envconf["N"]),
    #                         T                      =int(envconf["T"]),
    #                         with_visualization     =bool(int(envconf["WITH_VISUALIZATION"])),
    #                         framerate              =int(envconf["INIT_FRAMERATE"]),
    #                         print_enabled          =bool(int(envconf["PRINT_ENABLED"])),
    #                         plot_trajectory        =bool(int(envconf["PLOT_TRAJECTORY"])),
    #                         log_zarr_file          =bool(int(envconf["LOG_ZARR_FILE"])),
    #                         save_ext               =None,
    #                         agent_radius           =int(envconf["RADIUS_AGENT"]),
    #                         max_vel                =int(envconf["MAXIMUM_VELOCITY"]),
    #                         vis_field_res          =int(envconf["VISUAL_FIELD_RESOLUTION"]),
    #                         vision_range           =int(envconf["VISION_RANGE"]),
    #                         agent_fov              =float(envconf['AGENT_FOV']),
    #                         show_vision_range      =bool(int(envconf["SHOW_VISION_RANGE"])),
    #                         agent_consumption      =int(envconf["AGENT_CONSUMPTION"]),
    #                         N_res                  =int(envconf["N_RESOURCES"]),
    #                         patch_radius           =float(envconf["RADIUS_RESOURCE"]),
    #                         res_pos                =tuple(eval(envconf["RESOURCE_POS"])),
    #                         res_units              =tuple(eval(envconf["RESOURCE_UNITS"])),
    #                         res_quality            =tuple(eval(envconf["RESOURCE_QUALITY"])),
    #                         regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
    #                         landmark_radius        =int(envconf["RADIUS_LANDMARK"]),
    #                         NN                     =NN,
    #                         other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
    #                         vis_transform          =str(envconf["VIS_TRANSFORM"]),
    #                         percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
    #                         percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
    #                         action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
    #                         LM_dist_noise_std      =float(envconf["LM_DIST_NOISE_STD"]),
    #                         LM_angle_noise_std     =float(envconf["LM_ANGLE_NOISE_STD"]),
    #                         LM_radius_noise_std    =float(envconf["LM_RADIUS_NOISE_STD"]),
    #                         )
    #         t, dist, elapsed_time = sim.start()
    #         # print(f'Finished {load_dir}, runtime: {elapsed_time} sec, fitness: {t, d}')
    #     return t, dist


def reconstruct_NN(envconf,pv=None):
    """mirrors start_EA arch packaging"""
    
    # gather NN variables
    N                    = int(envconf["N"])

    # usual
    if N == 1:  num_class_elements = 4 # single-agent --> perception of 4 walls
    else:       num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes

    # overrides
    if 'nowalls' in envconf["SIM_TYPE"]: num_class_elements = 2 # multi-agent --> 2 agent modes
    elif 'social' in envconf["SIM_TYPE"]: num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes
    
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

    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep1'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep3'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep4'

    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep9'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep15'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep4'

    # exp_name = 'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep10'

    # exp_name = 'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep7'
    # exp_name = 'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep11'


    # exp_name = 'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep9' # vids - indiv nav
    # exp_name = 'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep4' # vids - follower
    # exp_name = 'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep37' # indiv nav

    # near-direct followers
    # exp_name = 'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep18' # vids - distracted
    # exp_name = 'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep11'
    # exp_name = 'sc_N5_NRW2_ND2_CNN14_FNN16_vis8_rep31'
    # exp_name = 'sc_N5_NRW1_ND3_CNN14_FNN16_vis8_rep15'

    # exp_name = 'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitAg100_rep22'
    # exp_name = 'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_SinitRes100_rep33'

    exp_name = 'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_SinitAg100_rep0'
    # exp_name = 'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_SinitAg100_rep4'



    gen_ext, valfit = find_top_val_gen(exp_name, 'cen')
    # NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NN0_pickle.bin'
    NN_pv_path = fr'{data_dir}/{exp_name}/{gen_ext}_NNcen_pickle.bin'
    env_path = fr'{data_dir}/{exp_name}/.env'

    with open(NN_pv_path,'rb') as f:
        pv = pickle.load(f)

    start(pv=pv, env_path=env_path, seed=300)
    # start()
