# ### ---- Simulate saved NN ---- ###
# import pickle

# # file_name = r"abm\data\NNs\50_128_2_avg572_chaotic_swarm"
# file_name = r"abm\data\simulation_data\roulette_51_128_1_sticky_T2000\gen29\NN0_fitness311\NN_pickle"
# # file_name = "crashed_NN"

# with open(f"{file_name}.bin", "rb") as f:
#     NN = pickle.load(f)

# # from abm.app import start
# # start(NN=NN)

# from abm.app import start_headless
# start_headless(NN=NN, sim_save_name='test')


# ### ---- Plot saved simulation ---- ###
# import os
# import zarr
# from abm.monitoring import plot_funcs

# sim_save_name = 'test'

# root_dir = os.path.dirname(os.path.realpath(__file__))
# save_dir = os.path.join(root_dir, 'abm\data\simulation_data', sim_save_name)
# print("save_dir: ", save_dir)

# ag_zarr = zarr.open(rf'{save_dir}\ag.zarr', mode='r')
# res_zarr = zarr.open(rf'{save_dir}\res.zarr', mode='r')
# plot_data = ag_zarr, res_zarr
# print("ag_zarr.shape: ", ag_zarr.shape)
# print("res_zarr.shape: ", res_zarr.shape)

# plot_funcs.plot_map(plot_data, x_max=400, y_max=400, save_dir=save_dir, save_name=sim_save_name)


### ---- EA trend plotting ---- ###
import pickle
import numpy as np
from abm.monitoring import plot_funcs

root_dir = 'abm\data\simulation_data'

# exp_folder = 'roulette_50_128_2_reflection_T4000'
# exp_folder = 'hybrid_50_128_2_reflection_T2000'
# exp_folder = 'hybrid_51_128_1_sticky_T2000'
# exp_folder = 'ES_51_128_1_sticky_T2000'
# exp_folder = 'roulette_51_128_1_sticky_T2000'
# exp_folder = 'plastic_hybrid_51_128_1_sticky_T2000'
# exp_folder = 'static_hybrid_51_128_1_sticky_T2000'
exp_folder = 'staticnoise_hybrid_51_128_1_sticky_T2000'

file_name = fr'{root_dir}\{exp_folder}\fitness_spread_per_generation'

with open(f"{file_name}.bin", "rb") as f:
    data_per_gen = pickle.load(f)
data_per_gen = np.array(data_per_gen)

save_dir = fr'{root_dir}\{exp_folder}'

plot_funcs.plot_EA_trend_violin(data_per_gen, save_dir)
plot_funcs.plot_EA_trend_violin(data_per_gen)


# ### ---- NN weights plotting ---- ###
# import pickle
# import torch
# import matplotlib.pyplot as plt

# file_name = r"abm\data\NNs\50_128_2_avg572_chaotic_swarm"
# with open(f"{file_name}.bin", "rb") as f:
#     NN = pickle.load(f)

# for m in NN.modules():
#     if isinstance(m, torch.nn.Linear):
#         print(m.weight.shape)
#         x = m.weight.detach().numpy()

#         print(type(m))

#         print(m.weight, type(m.weight))
#         print(m.weight.data, type(m.weight.data))
#         print(m.weight.detach().numpy(), type(m.weight.detach().numpy()))

#         plt.imshow(x)
#         plt.show()
