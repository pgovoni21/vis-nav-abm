# from abm.simulation.sims import Simulation
from abm.simulation.sims_target import Simulation

from contextlib import ExitStack
from pathlib import Path
from dotenv import dotenv_values
import os
import numpy as np

from abm.NN.model import WorldModel as Model

# calls env dict from root folder
env_path = Path(__file__).parent.parent / ".env"
envconf = dotenv_values(env_path)


def start(arch=None, pv=None, save_ext=None, seed=None): # "abm-start" in terminal

    # calls env dict from root folder
    env_path = Path(__file__).parent.parent / ".env"
    envconf = dotenv_values(env_path)

    # print(f'Running {save_ext}')

    # to run headless
    if int(envconf['WITH_VISUALIZATION']) == 0:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    # instantiate model object if called from EA
    NN = None
    if pv is not None:
        NN = Model(arch=arch, param_vector=pv)

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
                         CNN_depths             =[1,],
                         CNN_dims               =[4,], 
                         RNN_input_other_size   =int(envconf["RNN_INPUT_OTHER_SIZE"]),
                         RNN_hidden_size        =int(envconf["RNN_HIDDEN_SIZE"]),
                         LCL_output_size        =int(envconf["LCL_OUTPUT_SIZE"]),
                         NN_activ               =str(envconf["NN_ACTIVATION_FUNCTION"]),
                         )
        fitnesses, elapsed_time, crash = sim.start()

        # print(f'Finished {save_ext}, runtime: {elapsed_time} sec, fitness: {fitnesses[0]}, crashed? {crash}')

    return save_ext, fitnesses, elapsed_time, crash
