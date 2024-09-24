from abm.simulation.sims_target_pred import Simulation
from abm.NN.model_pred import Model

from contextlib import ExitStack
from pathlib import Path
import dotenv as de
import os
import pickle
import torch
import numpy as np
import random

def start(model_tuple=None, pv=None, load_dir=None, seed=None, mode='train', pv_h2o_act=None, T=100, x=None, y=None, display=False): # "abm-start" in terminal

    envconf = de.dotenv_values(load_dir / '.env')

    if pv is None:
        # vfr = int(envconf["VISUAL_FIELD_RESOLUTION"])
        vfr = 8
        o_size = vfr*2 + 1
        h_size = 50
        a_size = 8

        NN = Model(arch=(o_size, h_size, a_size), activ='relu', sharpness=1, param_vector=None, mode='train_pred')
        # x = 5
        # y = 400
        # with open(fr'{load_dir}/abm/data/simulation_data/CML_traj5000_step200_a8_s2000_sh0_nq0.0025_goalWwall_vfr8_patchintensive/model.bin', 'rb') as f:
        #     model = pickle.load(f)
        # NN = model

    else:
        arch, activ, sharpness = model_tuple
        NN = Model(arch=arch, activ=activ, sharpness=sharpness, param_vector=pv, mode='train_act')

    if pv_h2o_act is not None:
        NN.assign_params_h2o_act(pv_h2o_act)

    if display == False:
        envconf['WITH_VISUALIZATION'] = 0
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    else:
        envconf['WITH_VISUALIZATION'] = 1
        envconf['INIT_FRAMERATE'] = 100

    # Set seed according to EA parent function to circumvent multiprocessing bug
    np.random.seed(seed)

    with ExitStack():
        if x == None:
            sim = Simulation(env_size               =tuple(eval(envconf["ENV_SIZE"])),
                            window_pad             =int(envconf["WINDOW_PAD"]),
                            N                      =int(envconf["N"]),
                            T                      =T,
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
                            NN                     =NN,
                            other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
                            vis_transform          =str(envconf["VIS_TRANSFORM"]),
                            percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
                            percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
                            action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
                            mode                   =mode,
                            )
            results = sim.start()
        else:
            sim = Simulation(env_size               =tuple(eval(envconf["ENV_SIZE"])),
                            window_pad             =int(envconf["WINDOW_PAD"]),
                            N                      =int(envconf["N"]),
                            T                      =T,
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
                            NN                     =NN,
                            other_input            =int(envconf["RNN_OTHER_INPUT_SIZE"]),
                            vis_transform          =str(envconf["VIS_TRANSFORM"]),
                            percep_angle_noise_std =float(envconf["PERCEP_ANGLE_NOISE_STD"]),
                            percep_dist_noise_std  =float(envconf["PERCEP_DIST_NOISE_STD"]),
                            action_noise_std       =float(envconf["ACTION_NOISE_STD"]),
                            mode                   =mode,
                            x                      =x,
                            y                      =y,
                            )
            results = sim.start()

    return results



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def play_model(EA_save_name, num_steps):

    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, 'abm/data/simulation_data')
    EA_save_dir = Path(data_dir, EA_save_name)

    # with open(fr'{data_dir}/views.bin', 'rb') as f:
    #     views = pickle.load(f)
    views = None
    with open(Path(EA_save_dir, 'model.bin'), 'rb') as f:
        model = pickle.load(f)

    set_seed(0)
    _,_,_ = start(load_dir=EA_save_dir, NN=model, mode='test', T=num_steps, views=views, display=True)


if __name__ == '__main__':

    # name = 'CML_traj10000_step100_s2500_sh1_nq0.0025_nv0.0005_nw0.0005_goal108_patchintensive'

    # num_trajs=1000
    # num_steps=100
    # a_size=8
    # s_size=4000
    # sharpness=10
    # n_q=.0025
    # n_v=n_q/5
    # n_w=n_q/5
    # name = f'CML_traj{num_trajs}_step{num_steps}_a{a_size}_s{s_size}_sh{sharpness}_nq{n_q}_goalpatchW_vfr8_patchintensive'

    # play_model(name,1000)

    # set_seed(3)
    # start(load_dir=Path(__file__).parent.parent, mode='train', T=1000, display=True)


    num_gen = 1000
    pop_size = 50
    h_size = 100
    vis_trans = 'minmax'
    # vis_trans = 'maxWF'

    # start_EA_pred(num_gen, pop_size, h_size, vis_trans, 0, 'train_pred')

    # exp = 'pred_sepangdist_h500_a8_vis8_maxWF_gen10k_pop50_rep0'
    # gen = '9759'
    # exp = 'pred_sepangdist_h500_a8_relu_vis8_noWF_n1_rep0'
    # gen = '987'
    exp = 'pred_sepangdist_h100_a8_vis8_minmax_pop50_rep0'

    import os, pickle
    data_dir = Path(__file__).parent / fr'data/simulation_data/{exp}'

    gens = []
    for name in os.listdir(data_dir):
        if name.startswith('gen'):
            gen = int(''.join(filter(str.isdigit, name))[:-1])
            gens.append(gen)
    gens.sort()

    with open(fr'abm/data/simulation_data/{exp}/gen{gen}_NN0_pickle.bin', 'rb') as f:
        pv = pickle.load(f)

    arch = (33, 100, 8)
    activ = 'relu'
    sharpness = 1
    model_tuple = (arch, activ, sharpness)

    start(model_tuple, pv, Path(__file__).parent.parent, seed=0, mode='test', pv_h2o_act=None, T=100, display=True)

    