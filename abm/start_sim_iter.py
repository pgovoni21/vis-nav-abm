# from abm.simulation.sims import Simulation
# from abm.simulation.sims_target import Simulation
from abm.simulation.sims_target_iter import Simulation
# from abm.simulation.sims_target_double import Simulation
# from abm.simulation.sims_target_cross import Simulation

from pathlib import Path
import dotenv as de
import os
def start_env(model_tuple=None):

    envconf = de.dotenv_values(Path(__file__).parent.parent / '.env')

    envconf['WITH_VISUALIZATION'] = 0
    envconf['PLOT_TRAJECTORY'] = 0
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    global sim
    sim = Simulation(width                 =int(envconf["ENV_WIDTH"]),
                    height                 =int(envconf["ENV_HEIGHT"]),
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
                    min_res_perpatch       =int(envconf["MIN_RESOURCE_PER_PATCH"]),
                    max_res_perpatch       =int(envconf["MAX_RESOURCE_PER_PATCH"]),
                    min_res_quality        =float(envconf["MIN_RESOURCE_QUALITY"]),
                    max_res_quality        =float(envconf["MAX_RESOURCE_QUALITY"]),
                    regenerate_patches     =bool(int(envconf["REGENERATE_PATCHES"])),
                    NN                     =None,
                    model_tuple            =model_tuple,
                    )

def start_iter(pv=None, seed=None): 

    global sim

    fitnesses = sim.start(pv, seed)
    # print(f'Finished {save_ext}, runtime: {elapsed_time} sec, fitness: {int(fitnesses[0])}')

    return fitnesses
