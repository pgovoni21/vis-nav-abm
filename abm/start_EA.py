from abm.EA.EA import EvolAlgo

from pathlib import Path
from dotenv import dotenv_values


def start_EA(): # "EA-start" in terminal

    # calls env dict from root folder
    env_path = Path(__file__).parent.parent / ".env"
    envconf = dotenv_values(env_path)

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

    architecture = (
        CNN_input_size, 
        CNN_depths, 
        CNN_dims, 
        RNN_other_input_size, 
        RNN_hidden_size, 
        LCL_output_size
        )

    EA = EvolAlgo(arch                      =architecture, 
                  activ                     =str(envconf["NN_ACTIVATION_FUNCTION"]),
                  RNN_type                  =str(envconf["RNN_TYPE"]),
                  population_size           =int(envconf["EA_POPULATION_SIZE"]), 
                  init_sigma                =float(envconf["EA_INIT_SIGMA"]),
                  generations               =int(envconf["EA_GENERATIONS"]), 
                  episodes                  =int(envconf["EA_EPISODES"]), 
                  num_top_nn_saved          =int(envconf["EA_NUM_TOP_NN_SAVED"]),
                  num_top_nn_plots          =int(envconf["EA_NUM_TOP_NN_PLOTS"]),
                  EA_save_name              =str(envconf["EA_SAVE_NAME"]),
                  start_seed                =int(envconf["EA_START_SEED"]),
                  )
    EA.fit_parallel()


if __name__ == '__main__':
    start_EA()