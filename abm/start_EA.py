# from abm.metarunner.EA import EvolAlgo
from abm.metarunner.EA_PGPE import EvolAlgo
from abm.metarunner.EA_PGPE_pred import EvolAlgo as EvolAlgo_pred

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
    CNN_input_size       = (num_class_elements, vis_field_res)
    CNN_depths           = list(map(int,envconf["CNN_DEPTHS"].split(',')))
    CNN_dims             = list(map(int,envconf["CNN_DIMS"].split(',')))
    RNN_other_input_size = int(envconf["RNN_OTHER_INPUT_SIZE"])
    RNN_hidden_size      = int(envconf["RNN_HIDDEN_SIZE"])
    LCL_output_size      = int(envconf["LCL_OUTPUT_SIZE"])

    architecture = (
        CNN_input_size, 
        CNN_depths, 
        CNN_dims, 
        RNN_other_input_size, 
        RNN_hidden_size, 
        LCL_output_size,
        )

    EA = EvolAlgo(arch                      =architecture, 
                  activ                     =str(envconf["NN_ACTIVATION_FUNCTION"]),
                  RNN_type                  =str(envconf["RNN_TYPE"]),
                  population_size           =int(envconf["EA_POPULATION_SIZE"]), 
                  init_sigma                =float(envconf["EA_INIT_SIGMA"]),
                  step_sigma                =float(envconf["EA_STEP_SIGMA"]),
                  step_mu                   =float(envconf["EA_STEP_MU"]),
                  momentum                  =float(envconf["EA_MOMENTUM"]),
                  generations               =int(envconf["EA_GENERATIONS"]), 
                  episodes                  =int(envconf["EA_EPISODES"]), 
                  EA_save_name              =str(envconf["EA_SAVE_NAME"]),
                  start_seed                =int(envconf["EA_START_SEED"]),
                  est_method                =str(envconf["EA_EST_METHOD"])
                  )
    EA.fit_parallel()


def start_EA_pred(num_gen, pop_size, h_size, vis_trans, rep, mode, exp=None, gen=None): # "EA-start" in terminal

    # calls env dict from root folder
    env_path = Path(__file__).parent.parent / ".env"
    envconf = dotenv_values(env_path)

    vis_field_res = int(envconf["VISUAL_FIELD_RESOLUTION"])
    o_size = vis_field_res*2 + 1
    # h_size = 500
    a_size = 8
    arch = (o_size, h_size, a_size)

    # envconf['NN_ACTIVATION_FUNCTION'] = activ
    envconf["VIS_TRANSFORM"] == vis_trans

    envconf['EA_GENERATIONS'] = num_gen
    envconf['EA_POPULATION_SIZE'] = pop_size
    # envconf['EA_EPISODES'] = 20

    if exp is None:
        envconf['EA_SAVE_NAME'] = f'pred_sepangdist_h{h_size}_a8_vis8_{vis_trans}_pop{pop_size}_rep{rep}'
    else:
        envconf['EA_SAVE_NAME'] = exp

    if mode == 'train_pred':
        EA = EvolAlgo_pred(arch               =arch, 
                    activ                     =str(envconf["NN_ACTIVATION_FUNCTION"]),
                    sharpness                 =int(envconf["ACT_SHARPNESS"]),
                    population_size           =int(envconf["EA_POPULATION_SIZE"]), 
                    init_sigma                =float(envconf["EA_INIT_SIGMA"]),
                    step_sigma                =float(envconf["EA_STEP_SIGMA"]),
                    step_mu                   =float(envconf["EA_STEP_MU"]),
                    momentum                  =float(envconf["EA_MOMENTUM"]),
                    generations               =int(envconf["EA_GENERATIONS"]), 
                    episodes                  =int(envconf["EA_EPISODES"]), 
                    EA_save_name              =str(envconf["EA_SAVE_NAME"]),
                    start_seed                =int(envconf["EA_START_SEED"]),
                    est_method                =str(envconf["EA_EST_METHOD"]),
                    mode                      =mode,
                    )
    elif mode == 'train_act':
        EA = EvolAlgo_pred(arch               =arch, 
                    activ                     =str(envconf["NN_ACTIVATION_FUNCTION"]),
                    sharpness                 =int(envconf["ACT_SHARPNESS"]),
                    population_size           =int(envconf["EA_POPULATION_SIZE"]), 
                    init_sigma                =float(envconf["EA_INIT_SIGMA"]),
                    step_sigma                =float(envconf["EA_STEP_SIGMA"]),
                    step_mu                   =float(envconf["EA_STEP_MU"]),
                    momentum                  =float(envconf["EA_MOMENTUM"]),
                    generations               =int(envconf["EA_GENERATIONS"]), 
                    episodes                  =int(envconf["EA_EPISODES"]), 
                    EA_save_name              =str(envconf["EA_SAVE_NAME"]),
                    start_seed                =int(envconf["EA_START_SEED"]),
                    est_method                =str(envconf["EA_EST_METHOD"]),
                    mode                      =mode,
                    exp                       =exp,
                    gen                       =gen,
                    )
    EA.fit_parallel()


if __name__ == '__main__':
    start_EA()

    # num_gen = 1000
    # pop_size = 50
    # h_size = 100
    # vis_trans = 'minmax'
    # # vis_trans = 'maxWF'

    # # start_EA_pred(num_gen, pop_size, h_size, vis_trans, 0, 'train_pred')

    # # exp = 'pred_sepangdist_h500_a8_vis8_maxWF_gen10k_pop50_rep0'
    # # gen = '9759'
    # # exp = 'pred_sepangdist_h500_a8_relu_vis8_noWF_n1_rep0'
    # # gen = '987'
    # exp = 'pred_sepangdist_h100_a8_vis8_minmax_pop50_rep0'

    # import os
    # data_dir = Path(__file__).parent / fr'data/simulation_data/{exp}'

    # gens = []
    # for name in os.listdir(data_dir):
    #     if name.startswith('gen'):
    #         gen = int(''.join(filter(str.isdigit, name))[:-1])
    #         gens.append(gen)
    # gens.sort()

    # start_EA_pred(num_gen, pop_size, h_size, vis_trans, 0, 'train_act', exp, gens[-1])

    