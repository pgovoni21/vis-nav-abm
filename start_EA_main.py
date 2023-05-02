from abm.NN.EA import EvolAlgo

from pathlib import Path
from dotenv import dotenv_values

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

    EA = EvolAlgo(arch                      =architecture, 
                  activ                     =str(envconf["NN_ACTIVATION_FUNCTION"]),
                  dt                        =int(envconf["NN_DT"]), 
                  population_size           =int(envconf["EA_POPULATION_SIZE"]), 
                  generations               =int(envconf["EA_GENERATIONS"]), 
                  episodes                  =int(envconf["EA_EPISODES"]), 
                  num_top_saved             =int(envconf["EA_NUM_TOP_SAVED"]),
                  EA_save_name              =str(envconf["EA_SAVE_NAME"]),
                  start_seed                =int(envconf["EA_START_SEED"]),
                  )
    EA.fit_parallel()


if __name__ == '__main__':

    # calls env dict from root folder
    env_path = Path(__file__).parent / ".env"
    envconf = dotenv_values(env_path)

    start_EA()