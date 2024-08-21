from abm.simulation.sims_target_CML import Simulation

from contextlib import ExitStack
from pathlib import Path
import dotenv as de
import os
import pickle
import torch
import numpy as np
import random
from abm.NN.model_CML import Model

def start(load_dir=None, NN=None, mode='test', T=100, views=None, x=None, y=None, display=False): # "abm-start" in terminal

    envconf = de.dotenv_values(load_dir / '.env')

    if NN == None:
        vfr = int(envconf["VISUAL_FIELD_RESOLUTION"])
        # # onehot
        # with open(fr'{load_dir}/abm/data/simulation_data/views_vfr{vfr}.bin', 'rb') as f:
        #     views = pickle.load(f)
        # o_size = len(views)
        # for i,v in enumerate(views):
        #     print(f'{i}: {v}')
        # print(f'num views: {len(views)}')

        # vector
        o_size = vfr*4

        NN = Model(o_size=o_size, a_size=32, s_dim=1000, sharpness=1)
        # x = 5
        # y = 400
        # with open(fr'{load_dir}/abm/data/simulation_data/CML_traj5000_step200_a8_s2000_sh0_nq0.0025_goalWwall_vfr8_patchintensive/model.bin', 'rb') as f:
        #     model = pickle.load(f)
        # NN = model

    if display == False:
        envconf['WITH_VISUALIZATION'] = 0
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    else:
        envconf['WITH_VISUALIZATION'] = 1
        envconf['INIT_FRAMERATE'] = 100

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
                            views                  =views,
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
                            views                  =views,
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
    # name = 'CML_traj10000_step100_s2500_sh0.1_nq0.0025_nv0.0005_nw0.0005_goal108_patchintensive'
    # name = 'CML_traj10000_step100_s5000_sh0.1_nq0.0025_nv0.0005_nw0.0005_goal108_patchintensive'
    # name = 'CML_traj10000_step100_s5000_sh0.1_nq0.005_nv0.001_nw0.001_goal108_patchintensive'
    # name = 'CML_traj5000_step50_s4000_sh10_nq0.0025_nv0.0005_nw0.0005_goal108_patchintensive_a8'

    num_trajs=1000
    num_steps=100
    a_size=8
    s_size=4000
    sharpness=10
    n_q=.0025
    n_v=n_q/5
    n_w=n_q/5
    name = f'CML_traj{num_trajs}_step{num_steps}_a{a_size}_s{s_size}_sh{sharpness}_nq{n_q}_goalpatchW_vfr8_patchintensive'

    play_model(name,1000)

    # set_seed(3)
    # start(load_dir=Path(__file__).parent.parent, mode='train', T=1000, display=True)