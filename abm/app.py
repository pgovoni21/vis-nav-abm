from contextlib import ExitStack

from abm.simulation.sims import Simulation
# from abm.simulation.sims_current import Simulation_current
# from abm.simulation.isims import PlaygroundSimulation
# import abm.contrib.playgroundtool as pgt
from abm.NN.EA import EvolAlgo

import os
# loading env variables from dotenv file
from dotenv import dotenv_values

EXP_NAME = os.getenv("EXPERIMENT_NAME", "") # returns associated path if available, nothing if it doesn't exist
root_abm_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # moves up 2 levels (grandpa dir)
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env") # concatenates grandpa dir with env/exp file name
envconf = dotenv_values(env_path) # returns dict of this file

def start(NN=None, sim_save_name=None):
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
                         sim_save_name          =sim_save_name,
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
                         NN_weight_init         =None,
                         NN_input_other_size    =int(envconf["NN_INPUT_OTHER_SIZE"]),
                         NN_hidden_size         =int(envconf["NN_HIDDEN_SIZE"]),
                         NN_output_size         =int(envconf["NN_OUTPUT_SIZE"]),
                         )
        fitnesses, elapsed_time, crash = sim.start()
    return fitnesses, elapsed_time, crash

def start_headless(NN=None, sim_save_name=None):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    envconf['WITH_VISUALIZATION'] = 0
    fitnesses, elapsed_time, crash = start(NN=NN, sim_save_name=sim_save_name)
    return fitnesses, elapsed_time, crash


# def start_playground():
#     # changing env file according to playground default parameters before
#     # running any component of the SW
#     save_isims_env(root_abm_dir, EXP_NAME, pgt, envconf)
#     # Start interactive simulation
#     sim = PlaygroundSimulation()
#     sim.start()


# def save_isims_env(env_dir, _EXP_NAME, _pgt, _envconf):
#     """translating a default parameters dictionary to an environment
#     file and using env variable keys instead of class attribute names
#     :param env_dir: directory path of environemnt file"""
#     def_params = _pgt.default_params
#     def_env_vars = _pgt.def_env_vars
#     translator_dict = _pgt.def_params_to_env_vars
#     translated_dict = _envconf

#     for k in def_params.keys():
#         if k in list(translator_dict.keys()):
#             v = def_params[k]
#             if v == "True" or v is True:
#                 v = "1"
#             elif v == "False" or v is False:
#                 v = "0"
#             translated_dict[translator_dict[k]] = v
#     for def_env_name, def_env_val in def_env_vars.items():
#         translated_dict[def_env_name] = def_env_val

#     print("Saving playground default params in env file under path ", env_dir)
#     if os.path.isfile(os.path.join(env_dir, f"{_EXP_NAME}.env")):
#         os.remove(os.path.join(env_dir, f"{_EXP_NAME}.env"))
#     generate_env_file(translated_dict, f"{_EXP_NAME}.env", env_dir)

# def generate_env_file(env_data, file_name, save_folder):
#     """Generating a single env file under save_folder with file_name including env_data as env format"""
#     os.makedirs(save_folder, exist_ok=True)
#     file_path = os.path.join(save_folder, file_name)
#     with open(file_path, "a") as file:
#         for k, v in env_data.items():
#             file.write(f"{k}={v}\n")


# def start_current(parallel=False, agent_behave_param_list=None):
#     window_pad = 30
#     with ExitStack():
#         sim = Simulation_current(N=int(float(envconf["N"])),
#                          T=int(float(envconf["T"])),
#                          v_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
#                          agent_fov=float(envconf['AGENT_FOV']),
#                          framerate=int(float(envconf["INIT_FRAMERATE"])),
#                          with_visualization=bool(int(float(envconf["WITH_VISUALIZATION"]))),
#                          width=int(float(envconf["ENV_WIDTH"])),
#                          height=int(float(envconf["ENV_HEIGHT"])),
#                          show_vis_field=bool(int(float(envconf["SHOW_VISUAL_FIELDS"]))),
#                          show_vis_field_return=bool(int(envconf['SHOW_VISUAL_FIELDS_RETURN'])),
#                          pooling_time=int(float(envconf["POOLING_TIME"])),
#                          pooling_prob=float(envconf["POOLING_PROBABILITY"]),
#                          agent_radius=int(float(envconf["RADIUS_AGENT"])),
#                          N_resc=int(float(envconf["N_RESOURCES"])),
#                          allow_border_patch_overlap=bool(int(float(envconf["PATCH_BORDER_OVERLAP"]))),
#                          min_resc_perpatch=int(float(envconf["MIN_RESOURCE_PER_PATCH"])),
#                          max_resc_perpatch=int(float(envconf["MAX_RESOURCE_PER_PATCH"])),
#                          min_resc_quality=float(envconf["MIN_RESOURCE_QUALITY"]),
#                          max_resc_quality=float(envconf["MAX_RESOURCE_QUALITY"]),
#                          patch_radius=int(float(envconf["RADIUS_RESOURCE"])),
#                          regenerate_patches=bool(int(float(envconf["REGENERATE_PATCHES"]))),
#                          agent_consumption=int(float(envconf["AGENT_CONSUMPTION"])),
#                          ghost_mode=bool(int(float(envconf["GHOST_WHILE_EXPLOIT"]))),
#                          patchwise_exclusion=bool(int(float(envconf["PATCHWISE_SOCIAL_EXCLUSION"]))),
#                          teleport_exploit=bool(int(float(envconf["TELEPORT_TO_MIDDLE"]))),
#                          vision_range=int(float(envconf["VISION_RANGE"])),
#                          visual_exclusion=bool(int(float(envconf["VISUAL_EXCLUSION"]))),
#                          show_vision_range=bool(int(float(envconf["SHOW_VISION_RANGE"]))),
#                          use_ifdb_logging=bool(int(float(envconf["USE_IFDB_LOGGING"]))),
#                          use_ram_logging=bool(int(float(envconf["USE_RAM_LOGGING"]))),
#                          save_csv_files=bool(int(float(envconf["SAVE_CSV_FILES"]))),
#                          use_zarr=bool(int(float(envconf["USE_ZARR_FORMAT"]))),
#                          parallel=parallel,
#                          window_pad=window_pad,
#                          agent_behave_param_list=agent_behave_param_list,
#                          collide_agents=bool(int(float(envconf["AGENT_AGENT_COLLISION"])))
#                          )
#         sim.write_batch_size = 100
#         sim.start()

# def start_current_headless():
#     os.environ['SDL_VIDEODRIVER'] = 'dummy'
#     envconf['WITH_VISUALIZATION'] = '0'
#     start_current()


def start_EA():

    # calculate NN input size (visual/contact perception + other)
    N                   =int(envconf["N"])
    vis_field_res       =int(envconf["VISUAL_FIELD_RESOLUTION"])
    contact_field_res   =int(envconf["CONTACT_FIELD_RESOLUTION"])
    
    if N == 1:  num_class_elements = 4 # single-agent --> perception of 4 walls
    else:       num_class_elements = 6 # multi-agent --> perception of 4 walls + 2 agent modes

    vis_input_num = vis_field_res * num_class_elements
    contact_input_num = contact_field_res * num_class_elements
    other_input_num = int(envconf["NN_INPUT_OTHER_SIZE"]) # velocity + orientation + on_resrc
    
    # assemble NN architecture
    input_size = vis_input_num + contact_input_num + other_input_num
    hidden_size = int(envconf["NN_HIDDEN_SIZE"])
    output_size = int(envconf["NN_OUTPUT_SIZE"]) # dvel + dtheta
    architecture = (input_size, hidden_size, output_size)

    EA = EvolAlgo(architecture              =architecture, 
                  dt                        =int(envconf["EA_DISCRETIZATION_TIMESTEP"]), 
                  init                      =str(envconf["EA_NN_INIT_SCHEME"]), 
                  population_size           =int(envconf["EA_POPULATION_SIZE"]), 
                  generations               =int(envconf["EA_GENERATIONS"]), 
                  episodes                  =int(envconf["EA_EPISODES"]), 
                  mutation_variance         =float(envconf["EA_MUTATION_VARIANCE"]),
                  repop_method              =str(envconf["EA_REPOP_METHOD"]),
                  hybrid_scaling_factor     =float(envconf["EA_HYBRID_SCALING_FACTOR"]),
                  hybrid_new_intro_num      =int(envconf["EA_HYBRID_NEW_INTRO_NUM"]),
                  num_top_saved             =int(envconf["EA_NUM_TOP_SAVED"]),
                  EA_save_name              =str(envconf["EA_SAVE_NAME"]),
                  )
    EA.fit()