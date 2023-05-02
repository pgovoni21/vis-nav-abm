# ### ---- Simulate saved NN ---- ###
# import pickle

# root_dir = 'abm/data/simulation_data'

# exp_folder = 'stationarypatch_hybrid_50_128_1_sticky_T2000_DT10'

# NN_folder = 'gen8/NN0_af101'

# file_name = fr'{root_dir}/{exp_folder}/{NN_folder}/NN_pickle.bin'

# with open(file_name, 'rb') as f:
#     NN = pickle.load(f)

# from abm.app import start
# start(NN=NN)


# ### ---- Plot saved simulation ---- ###
# from pathlib import Path
# import zarr
# from abm.monitoring import plot_funcs

# sim_save_name = 'stationarypatch_ES_50_128_1_sticky_T2000/gen1/NN0_fitness5/ep1'

# root_dir = Path(__file__).parent.parent.parent
# save_dir = Path(root_dir, 'abm/data/simulation_data', sim_save_name)
# print("save_dir: ", save_dir)

# ag_zarr = zarr.open(fr'{save_dir}/ag.zarr', mode='r')
# res_zarr = zarr.open(fr'{save_dir}/res.zarr', mode='r')
# plot_data = ag_zarr, res_zarr
# print("ag_zarr.shape: ", ag_zarr.shape)
# print("res_zarr.shape: ", res_zarr.shape)

# plot_funcs.plot_map(plot_data, x_max=400, y_max=400)


# ### ---- EA trend plotting ---- ###
# import pickle
# import numpy as np
# from abm.monitoring import plot_funcs

# root_dir = 'abm/data/simulation_data'

# # exp_folder = 'static_hybrid_51_128_1_sticky_T2000'
# # exp_folder = 'staticnoise_hybrid_51_128_1_sticky_T2000'
# exp_folder = 'stationarypatch_hybrid_50_128_1_sticky_T2000_DT50'

# file_name = fr'{root_dir}/{exp_folder}/fitness_spread_per_generation'

# with open(f"{file_name}.bin", "rb") as f:
#     data_per_gen = pickle.load(f)

# save_dir = fr'{root_dir}/{exp_folder}'

# plot_funcs.plot_EA_trend_violin(data_per_gen, save_dir)
# plot_funcs.plot_EA_trend_violin(data_per_gen)


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
