import pickle
from abm.app import start
from abm.app import start_headless
import matplotlib.pyplot as plt
import torch

# file_name = "best_NN_gen7_avg571"
file_name = "crashed_NN"

with open(f"{file_name}.bin", "rb") as f:
    NN = pickle.load(f)

torch.set_printoptions(threshold=10_000)

# # start(NN=NN)
start_headless(NN=NN)



# file_name = "best_NN_avg_fitness_per_generation"

# with open(f"{file_name}.bin", "rb") as f:
#     data_per_gen = pickle.load(f)

# gens = range(len(data_per_gen))

# plt.plot(gens, data_per_gen)

# plt.show()



# for m in NN.modules():
#     if isinstance(m, torch.nn.Linear):
#         print(m.weight.shape)
#         x = m.weight.detach().numpy()
#         plt.imshow(x)
#         plt.show()