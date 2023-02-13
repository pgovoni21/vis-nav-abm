### ---- Simulate saved NN ---- ###
import pickle

file_name = r"abm\data\NNs\50_128_2_avg572_chaotic_swarm"
# file_name = "crashed_NN"

with open(f"{file_name}.bin", "rb") as f:
    NN = pickle.load(f)

# from abm.app import start
# start(NN=NN)

from abm.app import start_headless
start_headless(NN=NN, sim_save_name='test')


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


# ### ---- EA trend plotting ---- ###
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

# file_name = "fitness_spread_per_generation_50_128_2_roulette"
# with open(f"{file_name}.bin", "rb") as f:
#     data_per_gen = pickle.load(f)
# data_per_gen = np.array(data_per_gen)

# # num_gens, pop_size = data_per_gen.shape
# # for gen in range(num_gens):
# #     for NN in data_per_gen[:,gen]:
# #         plt.scatter(gen, NN)
# # plt.show()

# plt.imshow(data_per_gen)
# plt.show()


# ### ---- NN weights plotting ---- ###
# import pickle
# import torch
# import matplotlib.pyplot as plt

# file_name = "50_128_2_avg572_chaotic_swarm"
# with open(f"{file_name}.bin", "rb") as f:
#     NN = pickle.load(f)

# for m in NN.modules():
#     if isinstance(m, torch.nn.Linear):
#         print(m.weight.shape)
#         x = m.weight.detach().numpy()
#         plt.imshow(x)
#         plt.show()