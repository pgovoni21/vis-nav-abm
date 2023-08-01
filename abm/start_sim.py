# from abm.simulation.sims import Simulation
from abm.simulation.sims_target import Simulation

from contextlib import ExitStack
from pathlib import Path
import dotenv as de
import os
import numpy as np

from abm.NN.model import WorldModel as Model


def start(model_tuple=None, pv=None, save_ext=None, seed=None): # "abm-start" in terminal

    # calls env dict from root folder
    env_path = Path(__file__).parent.parent / ".env"
    envconf = de.dotenv_values(env_path)

    # print(f'Running {save_ext}')

    NN = None
    if pv is not None:
        arch, activ, RNN_type = model_tuple
        NN = Model(arch, activ, RNN_type, pv)

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
        fitnesses, elapsed_time, crash = sim.start()

        # print(f'Finished {save_ext}, runtime: {elapsed_time} sec, fitness: {fitnesses[0]}, crashed? {crash}')

    return save_ext, fitnesses, elapsed_time, crash


if __name__ == '__main__':

    # import pickle

    # data_dir = Path(__file__).parent / r'data/simulation_data'
    # name = r'stationarycorner_CNN12_FNN6_p25e5_sig0p1'
    # file_dir = Path(data_dir, name)

    # # compute individual PV from final distribution
    # with open(fr'{file_dir}/run_data.bin','rb') as f:
    #     mean_pv, std_pv, time = pickle.load(f)

    # final_mean_pv = mean_pv[-1,:]
    # final_std_pv = std_pv[-1,:]

    # indiv_vec = np.zeros(final_mean_pv.shape)
    # for n, (m,s) in enumerate(zip(final_mean_pv, final_std_pv)):
    #     indiv_vec[n] = np.random.normal(m,s)

    # # copy .env values + modify existing
    # envconf = de.dotenv_values(fr'{file_dir}/.env')
    # with open(Path(file_dir,'temp'), 'a') as file:
    #     for k, v in envconf.items():
    #         file.write(f"{k}={v}\n")
    
    start()