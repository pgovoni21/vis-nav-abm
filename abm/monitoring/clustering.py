import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import umap
import pacmap
import random

from pathlib import Path
import pickle
import dotenv as de
from abm.monitoring.trajs import find_top_val_gen, string_one_hot, agent_action_from_view, agent_action_from_xyo
from abm.start_sim import reconstruct_NN


def extract_parameters(names):
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    params_all = []
    for data_tuple, name in names:
    # for name in names:
        gen, valfit = find_top_val_gen(name, 'cen')
        with open(fr'{data_dir}/{name}/{gen}_NNcen_pickle.bin','rb') as f:
            params = pickle.load(f)
        params_all.append(params)
        # print(params.shape)
    params_all = np.array(params_all)
    return params_all


def extract_avgperfviews(names, space_step=5, orient_step=np.pi/256, data_type='spatial'):
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    # exp_name = names[0][1]
    exp_name = names[0]
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    width, height = tuple(eval(envconf["ENV_SIZE"]))
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    coll_boundary_thickness = int(envconf["RADIUS_AGENT"])
    x_range = np.linspace(x_min + coll_boundary_thickness, 
                        x_max - coll_boundary_thickness + 1, 
                        int((width - coll_boundary_thickness*2) / space_step+1))
    y_range = np.linspace(y_min + coll_boundary_thickness, 
                        y_max - coll_boundary_thickness + 1, 
                        int((height - coll_boundary_thickness*2) / space_step+1))

    # gather views
    basis_vfr = 8
    with open(fr'{data_dir}/IDM/views_vfr{basis_vfr}.bin', 'rb') as f:
        views = pickle.load(f)

    if data_type == 'spatial':
        # gather corresponding avg xyo
        avgxyori_per_view = np.zeros((len(views),3))
        for i,v in enumerate(views):
            view_onehot = string_one_hot(v)

            with open(fr'{data_dir}/IDM/IDM_transrot_c{space_step}_o{int(np.pi/orient_step)}_vsres{basis_vfr}_view{view_onehot}.bin', 'rb') as f:
                IDM = pickle.load(f)

            count_mat = IDM[:,:,0]
            ori_mat = IDM[:,:,1]
            nx,ny = count_mat.shape
            x,y,ori = zip(*[ (x_range[i], height-y_range[j], ori_mat[i,j]) for i in range(nx) for j in range(ny) if count_mat[i,j] == 0 ])
            avgori = np.arctan2(np.mean(np.sin(ori)), np.mean(np.cos(ori)))
            avgxyori_per_view[i] = np.array([np.mean(x), np.mean(y), avgori])
            # count = len(x)
            # print(f'view: {v, view_onehot} \t| average x/y/ori: {np.mean(x), np.mean(y), avgori, count}')

        # gather boundary_endpts
        width, height = tuple(eval(envconf["ENV_SIZE"]))
        x_min, x_max = 0, width
        y_min, y_max = 0, height
        boundary_endpts = [
                np.array([ x_min, y_min ]),
                np.array([ x_max, y_min ]),
                np.array([ x_min, y_max ]),
                np.array([ x_max, y_max ])
                ]

    elif data_type == 'spatial+social':
        views_social = views.copy() # set without viewable agent
        for view in views: # set with 1 explorer
            v = view.copy()
            x = np.random.randint(0,len(v))
            v[x] = 'agent_explore'
            views_social.append(v)
        for view in views: # set with 1 exploiter
            v = view.copy()
            x = np.random.randint(0,len(v))
            v[x] = 'agent_exploit'
            views_social.append(v)
        for view in views: # set with 2 explorer
            v = view.copy()
            x = np.random.randint(0,len(v)-1)
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            views_social.append(v)
        for view in views: # set with 2 exploiter
            v = view.copy()
            x = np.random.randint(0,len(v)-1)
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            views_social.append(v)
        for view in views: # set with 3 explorer
            v = view.copy()
            x = np.random.randint(0,len(v)-2)
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            views_social.append(v)
        for view in views: # set with 3 exploiter
            v = view.copy()
            x = np.random.randint(0,len(v)-2)
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            views_social.append(v)
        for view in views: # set with 4 explorer
            v = view.copy()
            x = np.random.randint(0,len(v)-3)
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            v[x+3] = 'agent_explore'
            views_social.append(v)
        for view in views: # set with 4 exploiter
            v = view.copy()
            x = np.random.randint(0,len(v)-3)
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            v[x+3] = 'agent_exploit'
            views_social.append(v)
        # views_social = random.sample(views_social, 100)
        views = views_social

    elif data_type == 'spatial+social x views':
        views_social = views.copy() # set without viewable agent
        for x in range(basis_vfr):
            for view in views: # set with 1 explorer
                v = view.copy()
                v[x] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr):
            for view in views: # set with 1 exploiter
                v = view.copy()
                v[x] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-1):
            for view in views: # set with 2 explorer
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-1):
            for view in views: # set with 2 exploiter
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-2):
            for view in views: # set with 3 explorer
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-2):
            for view in views: # set with 3 exploiter
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-3):
            for view in views: # set with 4 explorer
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                v[x+3] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-3):
            for view in views: # set with 4 exploiter
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                v[x+3] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-4): # set with 5 explorer x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                v[x+3] = 'agent_explore'
                v[x+4] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-4): # set with 5 exploiter x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                v[x+3] = 'agent_exploit'
                v[x+4] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-5): # set with 6 explorer x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                v[x+3] = 'agent_explore'
                v[x+4] = 'agent_explore'
                v[x+5] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-5): # set with 6 exploiter x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                v[x+3] = 'agent_exploit'
                v[x+4] = 'agent_exploit'
                v[x+5] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-6): # set with 7 explorer x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                v[x+3] = 'agent_explore'
                v[x+4] = 'agent_explore'
                v[x+5] = 'agent_explore'
                v[x+6] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-6): # set with 7 exploiter x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                v[x+3] = 'agent_exploit'
                v[x+4] = 'agent_exploit'
                v[x+5] = 'agent_exploit'
                v[x+6] = 'agent_exploit'
                views_social.append(v)
        for view in views:
            v = view.copy()
            v = ['agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore']
            views_social.append(v)
        for view in views:
            v = view.copy()
            v = ['agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit']
            views_social.append(v)
        views = views_social


    elif data_type == 'social x views':
        views_social = []
        for x in range(basis_vfr):
            for view in views: # set with 1 explorer
                v = view.copy()
                v[x] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr):
            for view in views: # set with 1 exploiter
                v = view.copy()
                v[x] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-1):
            for view in views: # set with 2 explorer
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-1):
            for view in views: # set with 2 exploiter
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-2):
            for view in views: # set with 3 explorer
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-2):
            for view in views: # set with 3 exploiter
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-3):
            for view in views: # set with 4 explorer
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                v[x+3] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-3):
            for view in views: # set with 4 exploiter
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                v[x+3] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-4): # set with 5 explorer x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                v[x+3] = 'agent_explore'
                v[x+4] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-4): # set with 5 exploiter x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                v[x+3] = 'agent_exploit'
                v[x+4] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-5): # set with 6 explorer x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                v[x+3] = 'agent_explore'
                v[x+4] = 'agent_explore'
                v[x+5] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-5): # set with 6 exploiter x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                v[x+3] = 'agent_exploit'
                v[x+4] = 'agent_exploit'
                v[x+5] = 'agent_exploit'
                views_social.append(v)
        for x in range(basis_vfr-6): # set with 7 explorer x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_explore'
                v[x+1] = 'agent_explore'
                v[x+2] = 'agent_explore'
                v[x+3] = 'agent_explore'
                v[x+4] = 'agent_explore'
                v[x+5] = 'agent_explore'
                v[x+6] = 'agent_explore'
                views_social.append(v)
        for x in range(basis_vfr-6): # set with 7 exploiter x each visual perturb
            for view in views:
                v = view.copy()
                v[x] = 'agent_exploit'
                v[x+1] = 'agent_exploit'
                v[x+2] = 'agent_exploit'
                v[x+3] = 'agent_exploit'
                v[x+4] = 'agent_exploit'
                v[x+5] = 'agent_exploit'
                v[x+6] = 'agent_exploit'
                views_social.append(v)
        for view in views:
            v = view.copy()
            v = ['agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore']
            views_social.append(v)
        for view in views:
            v = view.copy()
            v = ['agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit']
            views_social.append(v)
        views = views_social

    print(f'using {len(views)} # diff views/actions')

    # iterate over runs
    acts_all = []
    # for data_tuple, name in names:
    for name in names:
        gen, valfit = find_top_val_gen(name, 'cen')
        with open(fr'{data_dir}/{name}/{gen}_NNcen_pickle.bin','rb') as f:
            pv = pickle.load(f)
        envconf = de.dotenv_values(fr'{data_dir}/{name}/.env')
        NN, arch = reconstruct_NN(envconf, pv)
        # print(envconf["VISUAL_FIELD_RESOLUTION"])

        if data_type == 'spatial':
            acts = [agent_action_from_xyo(envconf, NN, boundary_endpts, x,y,ori) for x,y,ori in avgxyori_per_view] # use basis vfr 8 + iters over avg xyo
        elif data_type == 'spatial+social' or data_type == 'spatial+social x views' or data_type == 'social' or data_type == 'social-exploit' or data_type == 'social-explore':
            acts = [agent_action_from_view(envconf, NN, v) for v in views] # iters over spatial/social views (only works for single vfr)

        acts = np.abs(acts)
        acts_all.append(acts)
    
    return acts_all


def string_one_hot(view):
    onehot = ''
    for x in view:
        if x == 'wall_north': onehot += '0'
        elif x == 'wall_south': onehot += '1'
        elif x == 'wall_east': onehot += '2'
        elif x == 'wall_west': onehot += '3'
        else: print('invalid view')
    return onehot



def calculate_similarity_metrics(param_list, name_list):
    
    # Initialize result dictionaries
    num_models = len(param_list)
    cosine_sim = np.zeros((num_models, num_models))
    kl_div = np.zeros((num_models, num_models))
    wasserstein = np.zeros((num_models, num_models))
    euclidean_dist = np.zeros((num_models, num_models))
    
    # Calculate metrics for each pair of models
    for i in range(num_models):
        for j in range(num_models):
            params_i = param_list[i]
            params_j = param_list[j]
            
            # Cosine similarity
            norm_i = np.linalg.norm(params_i)
            norm_j = np.linalg.norm(params_j)
            if norm_i > 0 and norm_j > 0:
                cosine_sim[i, j] = np.dot(params_i, params_j) / (norm_i * norm_j)
            else:
                cosine_sim[i, j] = 0
            
            # KL divergence approximation using histograms
            # Normalize parameters for comparison
            params_i_norm = (params_i - np.mean(params_i)) / (np.std(params_i) + 1e-8)
            params_j_norm = (params_j - np.mean(params_j)) / (np.std(params_j) + 1e-8)
            
            # Create histograms with same bins
            bin_min = min(params_i_norm.min(), params_j_norm.min())
            bin_max = max(params_i_norm.max(), params_j_norm.max())
            bins = np.linspace(bin_min, bin_max, 100)
            
            hist_i, _ = np.histogram(params_i_norm, bins=bins, density=True)
            hist_j, _ = np.histogram(params_j_norm, bins=bins, density=True)
            
            # Add small epsilon to avoid division by zero
            hist_i = hist_i + 1e-10
            hist_j = hist_j + 1e-10
            
            # Normalize
            hist_i = hist_i / hist_i.sum()
            hist_j = hist_j / hist_j.sum()
            
            # KL divergence
            kl_div[i, j] = np.sum(hist_i * np.log(hist_i / hist_j))
            
            # Wasserstein distance (Earth Mover's Distance)
            wasserstein[i, j] = stats.wasserstein_distance(params_i_norm, params_j_norm)
            
            # Euclidean distance (normalized)
            euclidean_dist[i, j] = np.linalg.norm(params_i_norm - params_j_norm) / np.sqrt(len(params_i))
    
    # Calculate parameter statistics for each model
    param_stats = []
    for i, params in enumerate(param_list):
        stats_dict = {
            "model": name_list[i],
            "mean": np.mean(params),
            "std": np.std(params),
            "min": np.min(params),
            "max": np.max(params),
            "l2_norm": np.linalg.norm(params),
            "sparsity": np.sum(np.abs(params) < 1e-6) / len(params)
        }
        param_stats.append(stats_dict)
    
    return {
        "models": name_list,
        "param_vectors": param_list,
        "cosine_similarity": cosine_sim,
        "kl_divergence": kl_div,
        "wasserstein_distance": wasserstein,
        "euclidean_distance": euclidean_dist,
        "parameter_statistics": param_stats
    }


def visualize_similarity(similarity_results, class_types=None, data_label=None, annote=True, clustering=False, cluster_num=None, tsne_perplexity=30, umap_nbs=15, pacmap_params=None):
    """Visualize the similarity metrics between models using flattened parameters."""
    models = similarity_results["models"]
    num_models = len(models)
    param_vectors = similarity_results["param_vectors"]

    # Set up the figure
    if clustering: fig = plt.figure(figsize=(20, 12))
    else: fig = plt.figure(figsize=(20, 6))

    # Set up colors
    if class_types is not None:
        norm = class_types/(class_types.max()+1)
        if cluster_num == 3 or cluster_num == 6: colors = mpl.cm.hsv(norm)
        elif cluster_num == 2: colors = mpl.cm.plasma(norm)
        else: print(f'invalid cluster_num: {cluster_num}')
    elif data_label is not None:
        data_type, name_list = data_label
        if data_type == 'decorr_time':
            data = [data_tuple[1] for data_tuple, name in name_list]
        elif data_type == 'de_mean':
            data = [data_tuple[14] for data_tuple, name in name_list] # de_mean
        else: print('invalid data_type')
        data = np.array(data)
        norm = (data - data.min()) / (data.max() - data.min())
        colors = mpl.cm.plasma(norm)
    else:
        colors = mpl.cm.plasma(np.linspace(0, 1, num_models))

    alpha = 1

    # Plot 5: PCA scatter
    # # Prepare data (handle different vector lengths)
    # max_len = max(len(p) for p in param_vectors)
    # pca_data = np.zeros((num_models, max_len))
    # for i, params in enumerate(param_vectors):
    #     if len(params) < max_len:
    #         # Pad with zeros if shorter
    #         pca_data[i, :len(params)] = params
    #     else:
    #         # Sample uniformly if longer
    #         indices = np.linspace(0, len(params)-1, max_len, dtype=int)
    #         params = np.array(params)
    #         # print(i, pca_data.shape, params.shape, indices.shape)
    #         # pca_data[i] = params[indices]
    #         pca_data[i] = params

    # # Apply PCA
    # if clustering: ax1 = fig.add_subplot(231)
    # else: ax1 = fig.add_subplot(131)
    # pca_data = np.array(param_vectors)
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(pca_data)

    # ax1.scatter(pca_result[:, 0], pca_result[:, 1], s=50, c=colors, alpha=alpha)
    # if annote:
    #     for i, model_name in enumerate(models):
    #         ax1.annotate(model_name, (pca_result[i, 0], pca_result[i, 1]), fontsize=6)
    # ax1.set_title("PCA")
    # ax1.set_xlabel("PC1")
    # ax1.set_ylabel("PC2")
    # ax1.grid(True, alpha=0.3)


    # initial reduction before t-SNE/UMAP
    if np.array(param_vectors).shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        pca_result = pca.fit_transform(np.array(param_vectors))
        tsne_data = pca_result
        umap_data = pca_result
    else:
        tsne_data = np.array(param_vectors)
        umap_data = np.array(param_vectors)
    pacmap_data = np.array(param_vectors) # no pre-reduction, built-in

    # Apply t-SNE
    if clustering: ax2 = fig.add_subplot(231)
    else: ax2 = fig.add_subplot(131)
    tsne = TSNE(n_components=2, perplexity=min(tsne_perplexity, num_models-1), 
                max_iter=5000, random_state=42)
    tsne_result = tsne.fit_transform(tsne_data)

    ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], s=50, c=colors, alpha=alpha)
    if annote:
        for i, model_name in enumerate(models):
            ax2.annotate(model_name, (tsne_result[i, 0], tsne_result[i, 1]), 
                        fontsize=6, ha='center')
    ax2.set_title("t-SNE")
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    ax2.grid(True, alpha=0.3)

    # Plot 6: UMAP
    if clustering: ax3 = fig.add_subplot(232)
    else: ax3 = fig.add_subplot(132)
    reducer = umap.UMAP(n_neighbors=umap_nbs, min_dist=0.1, n_components=2, random_state=42)
    umap_result = reducer.fit_transform(umap_data)

    ax3.scatter(umap_result[:, 0], umap_result[:, 1], s=50, c=colors, alpha=alpha)
    if annote:
        for i, model_name in enumerate(models):
            ax3.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), fontsize=6)
    ax3.set_title("UMAP")
    ax3.set_xlabel("Dimension 1")
    ax3.set_ylabel("Dimension 2")
    ax3.grid(True, alpha=0.3)

    # Plot 6: PaCMAP
    nbs, mn, fp = pacmap_params
    if clustering: ax1 = fig.add_subplot(233)
    else: ax1 = fig.add_subplot(133)
    reducer = pacmap.PaCMAP(n_neighbors=nbs, MN_ratio=mn, FP_ratio=fp, n_components=2, apply_pca=True)
    pacmap_result = reducer.fit_transform(pacmap_data, init='pca')

    ax1.scatter(pacmap_result[:, 0], pacmap_result[:, 1], s=50, c=colors, alpha=alpha)
    if annote:
        for i, model_name in enumerate(models):
            ax1.annotate(model_name, (pacmap_result[i, 0], pacmap_result[i, 1]), fontsize=6)
    ax1.set_title("PaCMAP")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.grid(True, alpha=0.3)

    if clustering:
        from sklearn.mixture import GaussianMixture
        ax5 = fig.add_subplot(234)
        model = GaussianMixture(n_components=cluster_num, n_init=50, random_state=42)
        clusters = model.fit_predict(tsne_result)
        centers = model.means_
        ax5.scatter(tsne_result[:,0], tsne_result[:,1], c=clusters, cmap='plasma', alpha=alpha)
        ax5.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=alpha)
        ax5.set_title("t-SNE + GMM")
        ax5.set_xlabel("Dimension 1")
        ax5.set_ylabel("Dimension 2")
        ax5.grid(True, alpha=0.3)

        unique, counts = np.unique(clusters, return_counts=True)
        print('tSNE: ', dict(zip(unique, counts)))

        ax6 = fig.add_subplot(235)
        model = GaussianMixture(n_components=cluster_num, n_init=50, random_state=42)
        clusters = model.fit_predict(umap_result)
        centers = model.means_
        ax6.scatter(umap_result[:,0], umap_result[:,1], c=clusters, cmap='plasma', alpha=alpha)
        ax6.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=alpha)
        ax6.set_title("UMAP + GMM")
        ax6.set_xlabel("Dimension 1")
        ax6.set_ylabel("Dimension 2")
        ax6.grid(True, alpha=0.3)

        unique, counts = np.unique(clusters, return_counts=True)
        print('UMAP: ', dict(zip(unique, counts)))

        ax4 = fig.add_subplot(236)
        model = GaussianMixture(n_components=cluster_num, n_init=50, random_state=42)
        clusters = model.fit_predict(pacmap_result)
        centers = model.means_
        ax4.scatter(pacmap_result[:,0], pacmap_result[:,1], c=clusters, cmap='plasma', alpha=alpha)
        ax4.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=alpha)
        ax4.set_title("PaCMAP + GMM")
        ax4.set_xlabel("Dimension 1")
        ax4.set_ylabel("Dimension 2")
        ax4.grid(True, alpha=0.3)

        unique, counts = np.unique(clusters, return_counts=True)
        print('PaCMAP: ', dict(zip(unique, counts)))

    # cbars
    axs = (ax1, ax2, ax3)
    if class_types is not None:

        if cluster_num == 3: 
            cmap = mpl.cm.hsv
            bounds = np.linspace(0,1,6+1)
            tick_pos = bounds[:-1] + np.diff(bounds)/2
            num_pts = int(cmap.N * 6/7)
            norm = mpl.colors.BoundaryNorm(bounds, num_pts)
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, 
                                label='Class Types', fraction=0.046, pad=0.04)
            cbar.set_ticks(ticks=tick_pos, labels=['IS', 'IS/DP', 'DP', 'DP/BD', 'BD', 'BD/IS'])
            cbar.solids.set(alpha=.7)
        elif cluster_num == 2: 
            cmap = mpl.cm.plasma
            bounds = np.linspace(0,1,3+1)
            tick_pos = bounds[:-1] + np.diff(bounds)/2
            num_pts = int(cmap.N * 3/4)
            norm = mpl.colors.BoundaryNorm(bounds, num_pts)
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, 
                                label='Class Types', fraction=0.046, pad=0.04)
            cbar.set_ticks(ticks=tick_pos, labels=['IS', 'BD/IS', 'BD'])
            cbar.solids.set(alpha=.7)
        elif cluster_num == 6: 
            cmap = mpl.cm.hsv
            bounds = np.linspace(0,1,6+1)
            tick_pos = bounds[:-1] + np.diff(bounds)/2
            num_pts = int(cmap.N * 6/7)
            norm = mpl.colors.BoundaryNorm(bounds, num_pts)
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, 
                                label='Class Types', fraction=0.046, pad=0.04)
            cbar.set_ticks(ticks=tick_pos, labels=['F', 'F-mix', 'bw', 'IN-mix', 'IN', 'IN-F'])
            cbar.solids.set(alpha=.7)
        else: print(f'invalid cluster_num: {cluster_num}')
    elif data_label is not None:
        if data_type == 'decorr_time':
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(data.min(), data.max()), cmap='plasma'), ax=axs,
                        label='Decorrelation Time', fraction=0.046, pad=0.04)
        elif data_type == 'de_mean':
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(data.min(), data.max()), cmap='plasma'), ax=axs,
                        label='Directional Entropy', fraction=0.046, pad=0.04)
        cbar.solids.set(alpha=.7)
    else:
        plt.tight_layout()

    # if clustering:
    #     axs = (ax5, ax6)
    #     cmap = mpl.cm.plasma
    #     bounds = np.linspace(0,1,3+1)
    #     tick_pos = bounds[:-1] + np.diff(bounds)/2
    #     num_pts = int(cmap.N * 3/4)
    #     norm = mpl.colors.BoundaryNorm(bounds, num_pts)
    #     cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, 
    #                         label='Clusters', fraction=0.046, pad=0.04)
    #     cbar.set_ticks(ticks=tick_pos)
    #     cbar.solids.set(alpha=.7)

    return fig


def visualize_similarity_tsne_perp(similarity_results, class_types=None, data_label=None):
    """Visualize the similarity metrics between models using flattened parameters."""
    models = similarity_results["models"]
    num_models = len(models)
    param_vectors = similarity_results["param_vectors"]

    # initial reduction
    if len(param_vectors[0]) > 50:
        pca = PCA(n_components=50, random_state=42)
        tsne_data = pca.fit_transform(np.array(param_vectors))
    else:
        tsne_data = np.array(param_vectors)

    if class_types is not None:
        norm = class_types/(class_types.max()+1)
        colors = mpl.cm.hsv(norm)
        alpha = .7
        size = 50
    elif data_label is not None:
        data_type, name_list = data_label
        if data_type == 'decorr_time':
            data = [data_tuple[1] for data_tuple, name in name_list]
        elif data_type == 'de_mean':
            data = [data_tuple[14] for data_tuple, name in name_list] # de_mean
        else: print('invalid data_type')
        data = np.array(data)
        norm = (data - data.min()) / (data.max() - data.min())
        colors = mpl.cm.plasma(norm)
        alpha = .7
        size = 50
    else:
        colors = mpl.cm.plasma(np.linspace(0, 1, num_models))
        alpha = 1
        size = 100

    fig = plt.figure(figsize=(20, 12))

    ax1 = fig.add_subplot(231)
    tsne = TSNE(n_components=2, perplexity=5, 
                max_iter=5000, random_state=42)
    tsne_result = tsne.fit_transform(tsne_data)
    ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax1.annotate(model_name, (tsne_result[i, 0], tsne_result[i, 1]), 
                    fontsize=6, ha='center')
    ax1.set_title("Perplexity = 5", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Dimension 1", fontsize=12)
    ax1.set_ylabel("Dimension 2", fontsize=12)

    ax2 = fig.add_subplot(232)
    tsne = TSNE(n_components=2, perplexity=10, 
                max_iter=5000, random_state=42)
    tsne_result = tsne.fit_transform(tsne_data)
    ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax2.annotate(model_name, (tsne_result[i, 0], tsne_result[i, 1]), 
                    fontsize=6, ha='center')
    ax2.set_title("Perplexity = 10", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Dimension 1", fontsize=12)
    ax2.set_ylabel("Dimension 2", fontsize=12)

    ax3 = fig.add_subplot(233)
    tsne = TSNE(n_components=2, perplexity=20, 
                max_iter=5000, random_state=42)
    tsne_result = tsne.fit_transform(tsne_data)
    ax3.scatter(tsne_result[:, 0], tsne_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax3.annotate(model_name, (tsne_result[i, 0], tsne_result[i, 1]), 
                    fontsize=6, ha='center')
    ax3.set_title("Perplexity = 20", fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Dimension 1", fontsize=12)
    ax3.set_ylabel("Dimension 2", fontsize=12)

    ax4 = fig.add_subplot(234)
    tsne = TSNE(n_components=2, perplexity=30, 
                max_iter=5000, random_state=42)
    tsne_result = tsne.fit_transform(tsne_data)
    ax4.scatter(tsne_result[:, 0], tsne_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax4.annotate(model_name, (tsne_result[i, 0], tsne_result[i, 1]), 
                    fontsize=6, ha='center')
    ax4.set_title("Perplexity = 30", fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Dimension 1", fontsize=12)
    ax4.set_ylabel("Dimension 2", fontsize=12)

    ax5 = fig.add_subplot(235)
    tsne = TSNE(n_components=2, perplexity=40, 
                max_iter=5000, random_state=42)
    tsne_result = tsne.fit_transform(tsne_data)
    ax5.scatter(tsne_result[:, 0], tsne_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax5.annotate(model_name, (tsne_result[i, 0], tsne_result[i, 1]), 
                    fontsize=6, ha='center')
    ax5.set_title("Perplexity = 40", fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel("Dimension 1", fontsize=12)
    ax5.set_ylabel("Dimension 2", fontsize=12)

    ax6 = fig.add_subplot(236)
    tsne = TSNE(n_components=2, perplexity=49, 
                max_iter=5000, random_state=42)
    tsne_result = tsne.fit_transform(tsne_data)
    ax6.scatter(tsne_result[:, 0], tsne_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax6.annotate(model_name, (tsne_result[i, 0], tsne_result[i, 1]), 
                    fontsize=6, ha='center')
    ax6.set_title("Perplexity = 49", fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlabel("Dimension 1", fontsize=12)
    ax6.set_ylabel("Dimension 2", fontsize=12)
    
    plt.tight_layout()
    return fig


def visualize_similarity_umap_perp(similarity_results, class_types=None, data_label=None):
    """Visualize the similarity metrics between models using flattened parameters."""
    models = similarity_results["models"]
    num_models = len(models)
    param_vectors = similarity_results["param_vectors"]

    # initial reduction
    if len(param_vectors[0]) > 50:
        pca = PCA(n_components=50, random_state=42)
        umap_data = pca.fit_transform(np.array(param_vectors))
    else:
        umap_data = np.array(param_vectors)

    if class_types is not None:
        norm = class_types/(class_types.max()+1)
        colors = mpl.cm.hsv(norm)
        alpha = .7
        size = 50
    elif data_label is not None:
        data_type, name_list = data_label
        if data_type == 'decorr_time':
            data = [data_tuple[1] for data_tuple, name in name_list]
        elif data_type == 'de_mean':
            data = [data_tuple[14] for data_tuple, name in name_list] # de_mean
        else: print('invalid data_type')
        data = np.array(data)
        norm = (data - data.min()) / (data.max() - data.min())
        colors = mpl.cm.plasma(norm)
        alpha = .7
        size = 50
    else:
        colors = mpl.cm.plasma(np.linspace(0, 1, num_models))
        alpha = 1
        size = 100

    fig = plt.figure(figsize=(20, 12))

    ax1 = fig.add_subplot(231)
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, random_state=42)
    umap_result = reducer.fit_transform(umap_data)
    ax1.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax1.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax1.set_title("# Neighbors = 5", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Dimension 1", fontsize=12)
    ax1.set_ylabel("Dimension 2", fontsize=12)

    ax2 = fig.add_subplot(232)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
    umap_result = reducer.fit_transform(umap_data)
    ax2.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax2.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax2.set_title("# Neighbors = 10", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Dimension 1", fontsize=12)
    ax2.set_ylabel("Dimension 2", fontsize=12)

    ax3 = fig.add_subplot(233)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_result = reducer.fit_transform(umap_data)
    ax3.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax3.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax3.set_title("# Neighbors = 15", fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Dimension 1", fontsize=12)
    ax3.set_ylabel("Dimension 2", fontsize=12)

    ax4 = fig.add_subplot(234)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
    umap_result = reducer.fit_transform(umap_data)
    ax4.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax4.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax4.set_title("# Neighbors = 20", fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Dimension 1", fontsize=12)
    ax4.set_ylabel("Dimension 2", fontsize=12)

    ax5 = fig.add_subplot(235)
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42)
    umap_result = reducer.fit_transform(umap_data)
    ax5.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax5.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax5.set_title("# Neighbors = 50", fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel("Dimension 1", fontsize=12)
    ax5.set_ylabel("Dimension 2", fontsize=12)

    ax6 = fig.add_subplot(236)
    reducer = umap.UMAP(n_neighbors=100, min_dist=0.1, n_components=2, random_state=42)
    umap_result = reducer.fit_transform(umap_data)
    ax6.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax6.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax6.set_title("# Neighbors = 100", fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlabel("Dimension 1", fontsize=12)
    ax6.set_ylabel("Dimension 2", fontsize=12)
    
    plt.tight_layout()
    return fig



def visualize_similarity_pacmap_perp(similarity_results, class_types=None, data_label=None, init='random', mn=0.5, fp=2):
    """Visualize the similarity metrics between models using flattened parameters."""
    models = similarity_results["models"]
    num_models = len(models)
    param_vectors = similarity_results["param_vectors"]
    data = np.array(param_vectors)

    if class_types is not None:
        norm = class_types/(class_types.max()+1)
        colors = mpl.cm.hsv(norm)
        alpha = .7
        size = 50
    elif data_label is not None:
        data_type, name_list = data_label
        if data_type == 'decorr_time':
            data = [data_tuple[1] for data_tuple, name in name_list]
        elif data_type == 'de_mean':
            data = [data_tuple[14] for data_tuple, name in name_list] # de_mean
        else: print('invalid data_type')
        data = np.array(data)
        norm = (data - data.min()) / (data.max() - data.min())
        colors = mpl.cm.plasma(norm)
        alpha = .7
        size = 50
    else:
        colors = mpl.cm.plasma(np.linspace(0, 1, num_models))
        alpha = 1
        size = 100

    fig = plt.figure(figsize=(20, 12))

    ax1 = fig.add_subplot(231)
    reducer = pacmap.PaCMAP(n_neighbors=5, MN_ratio=mn, FP_ratio=fp, n_components=2)
    umap_result = reducer.fit_transform(data, init=init)
    ax1.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax1.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax1.set_title("# Neighbors = 5", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Dimension 1", fontsize=12)
    ax1.set_ylabel("Dimension 2", fontsize=12)

    ax2 = fig.add_subplot(232)
    reducer = pacmap.PaCMAP(n_neighbors=10, MN_ratio=mn, FP_ratio=fp, n_components=2)
    umap_result = reducer.fit_transform(data, init=init)
    ax2.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax2.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax2.set_title("# Neighbors = 10", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Dimension 1", fontsize=12)
    ax2.set_ylabel("Dimension 2", fontsize=12)

    ax3 = fig.add_subplot(233)
    reducer = pacmap.PaCMAP(n_neighbors=15, MN_ratio=mn, FP_ratio=fp, n_components=2)
    umap_result = reducer.fit_transform(data, init=init)
    ax3.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax3.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax3.set_title("# Neighbors = 15", fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Dimension 1", fontsize=12)
    ax3.set_ylabel("Dimension 2", fontsize=12)

    ax4 = fig.add_subplot(234)
    reducer = pacmap.PaCMAP(n_neighbors=20, MN_ratio=mn, FP_ratio=fp, n_components=2)
    umap_result = reducer.fit_transform(data, init=init)
    ax4.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax4.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax4.set_title("# Neighbors = 20", fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel("Dimension 1", fontsize=12)
    ax4.set_ylabel("Dimension 2", fontsize=12)

    ax5 = fig.add_subplot(235)
    reducer = pacmap.PaCMAP(n_neighbors=50, MN_ratio=mn, FP_ratio=fp, n_components=2)
    umap_result = reducer.fit_transform(data, init=init)
    ax5.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax5.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax5.set_title("# Neighbors = 50", fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel("Dimension 1", fontsize=12)
    ax5.set_ylabel("Dimension 2", fontsize=12)

    ax6 = fig.add_subplot(236)
    reducer = pacmap.PaCMAP(n_neighbors=100, MN_ratio=mn, FP_ratio=fp, n_components=2)
    umap_result = reducer.fit_transform(data, init=init)
    ax6.scatter(umap_result[:, 0], umap_result[:, 1], s=size, c=colors, alpha=alpha)
    for i, model_name in enumerate(models):
        ax6.annotate(model_name, (umap_result[i, 0], umap_result[i, 1]), 
                    fontsize=6, ha='center')
    ax6.set_title("# Neighbors = 100", fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlabel("Dimension 1", fontsize=12)
    ax6.set_ylabel("Dimension 2", fontsize=12)
    
    plt.tight_layout()
    return fig


def visualize_similarity_tables(similarity_results, class_borders=None):
    """Visualize the similarity metrics between models using flattened parameters."""
    models = similarity_results["models"]
    param_vectors = similarity_results["param_vectors"]

    # Set up the figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("IS | ISDP | DP | DPBD | BD | BDIS", fontsize=14)
    
    # Plot 1: Heatmap of cosine similarity
    ax1 = fig.add_subplot(231)
    sns.heatmap(similarity_results["cosine_similarity"], 
                annot=False, cmap="YlGnBu", vmin=-1, vmax=1,
                xticklabels=20, yticklabels=20)
    if class_borders is not None: 
        ax1.hlines(class_borders, *ax1.get_xlim(), color='k', linewidth=1)
        ax1.vlines(class_borders, *ax1.get_ylim(), color='k', linewidth=1)
    ax1.set_title("Cosine Similarity")

    # Plot 2: Heatmap of KL divergence
    ax2 = fig.add_subplot(232)
    sns.heatmap(similarity_results["kl_divergence"], 
                annot=False, cmap="YlOrRd",
                xticklabels=20, yticklabels=20)
    if class_borders is not None: 
        ax2.hlines(class_borders, *ax2.get_xlim(), color='k', linewidth=1)
        ax2.vlines(class_borders, *ax2.get_ylim(), color='k', linewidth=1)
    ax2.set_title("KL Divergence")

    # Plot 3: Heatmap of Wasserstein distance
    ax3 = fig.add_subplot(233)
    sns.heatmap(similarity_results["wasserstein_distance"], 
                annot=False, cmap="YlOrRd",
                xticklabels=20, yticklabels=20)
    if class_borders is not None: 
        ax3.hlines(class_borders, *ax3.get_xlim(), color='k', linewidth=1)
        ax3.vlines(class_borders, *ax3.get_ylim(), color='k', linewidth=1)
    ax3.set_title("Wasserstein Distance")


    # Plot 3: Heatmap of Euclidean distance
    ax4 = fig.add_subplot(234)
    sns.heatmap(similarity_results["euclidean_distance"], 
                annot=False, cmap="YlOrRd",
                xticklabels=20, yticklabels=20)
    if class_borders is not None: 
        ax4.hlines(class_borders, *ax3.get_xlim(), color='k', linewidth=1)
        ax4.vlines(class_borders, *ax3.get_ylim(), color='k', linewidth=1)
    ax4.set_title("Euclidean Distance")

    
    # Plot 4: Parameter distribution comparison
    ax5 = fig.add_subplot(235)
    # for i, params in enumerate(param_vectors):
    #     sns.kdeplot(params, label=models[i], ax=ax6)
    all_data = -np.array(param_vectors).flatten() # flip pos/neg if action
    sns.kdeplot(all_data, label='All', ax=ax5, color='k')
    # ax6.vlines([np.median(all_data)], *ax6.get_ylim(), color='r')
    ax5.set_title("KDE")
    ax5.set_xlabel("Value")
    ax5.set_ylabel("Density")
    # ax6.legend()

    
    # Plot 6: Parameter statistics comparison
    ax6 = fig.add_subplot(236)
    stats_data = similarity_results["parameter_statistics"]

    # Convert to format suitable for plotting
    stats_for_plot = {
        "Model": [],
        "Metric": [],
        "Value": []
    }
    
    metrics_to_plot = ["mean", "std", "l2_norm", "sparsity"]
    for stat_dict in stats_data:
        for metric in metrics_to_plot:
            stats_for_plot["Model"].append(stat_dict["model"])
            stats_for_plot["Metric"].append(metric)
            stats_for_plot["Value"].append(stat_dict[metric])
    
    # Plot as grouped bar chart
    sns.barplot(x="Metric", y="Value", hue="Model", data=stats_for_plot, ax=ax6)
    ax6.set_title("Parameter Statistics Comparison")
    ax6.legend(title="Model")

    return fig


def gamut_dict_sort(group):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'

    if group == 'all' or group == 'main_fig':
        with open(fr'{data_dir}/traj_matrices/gamut_visall_nodist_labeled.bin', 'rb') as f:
            data1 = pickle.load(f)
        with open(fr'{data_dir}/traj_matrices/gamut_vis8_dist_labeled.bin', 'rb') as f:
            data2 = pickle.load(f)
        if group == 'all':
            with open(fr'{data_dir}/traj_matrices/gamut_vis16_dist_labeled.bin', 'rb') as f:
                data3 = pickle.load(f)
            with open(fr'{data_dir}/traj_matrices/gamut_vis32_dist_labeled.bin', 'rb') as f:
                data4 = pickle.load(f)
            data = data1 | data2 | data3 | data4
        else:
            data = data1 | data2
        print(f'data dict len: {len(data)}')

        IS = []
        ISDP = []
        DP = []
        DPBD = []
        BD = []
        BDIS = []

        for name in data.keys():

            data_tuple, label = data[name]

            if label == 'IS':
                IS.append((data_tuple, name))
            elif label == 'IS/DP':
                ISDP.append((data_tuple, name))
            elif label == 'DP':
                DP.append((data_tuple, name))
            elif label == 'DP/BD':
                DPBD.append((data_tuple, name))
            elif label == 'BD':
                BD.append((data_tuple, name))
            elif label == 'BD/IS':
                BDIS.append((data_tuple, name))
            else:
                print(label)

    return IS, ISDP, DP, DPBD, BD, BDIS


def indivperturb_data(names, metric='median', types='all'):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # # init plot details
    fig, ax1 = plt.subplots(figsize=(6,4)) 

    # iterate over each file
    data_all = []
    for num, name in enumerate(names):

        data_indiv = []

        filename = 'val_matrix_cen'
        with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
            data = pickle.load(f)
        if metric == 'median': data_indiv.append(np.median(data))
        elif metric == 'num_found': data_indiv.append((data<1000).sum())

        if types == 'all':
            filename = 'val_matrix_cen_allRW_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_allD_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_nosocial_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                if metric == 'median': data_indiv.append(np.median(data))
                elif metric == 'num_found': data_indiv.append((data<1000).sum())
            filename = 'val_matrix_cen_selfsocial_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_ghostexplorer_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_ghostexploiter_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_N2-ghostexplorer_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
            filename = 'val_matrix_cen_N2-ghostexploiter_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                data_indiv.append(np.median(data))
        elif types == 'nosocial':
            filename = 'val_matrix_cen_nosocial_perturb'
            if Path(fr'{data_dir}/{name}/{filename}.bin').is_file():
                with open(fr'{data_dir}/{name}/{filename}.bin','rb') as f:
                    data = pickle.load(f)
                if metric == 'median': data_indiv.append(np.median(data))
                elif metric == 'num_found': data_indiv.append((data<1000).sum())
        
        data_all.append(data_indiv)

    data = np.array(data_all)
    # print(data.shape) # num_runs, num_perturbs = data.shape

    return data


def extract_acts(names, data_type='abs(action)'):
    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_name = names[0]
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)

    # gather views
    basis_vfr = 8
    with open(fr'{data_dir}/IDM/views_vfr{basis_vfr}.bin', 'rb') as f:
        views = pickle.load(f)
    num_views = len(views)

    views_list = views.copy() # start with set without viewable agent

    for x in range(basis_vfr): # set with 1 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr): # set with 1 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            views_list.append(v)
    for x in range(basis_vfr-1): # set with 2 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-1): # set with 2 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            views_list.append(v)
    for x in range(basis_vfr-2): # set with 3 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-2): # set with 3 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            views_list.append(v)
    for x in range(basis_vfr-3): # set with 4 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            v[x+3] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-3): # set with 4 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            v[x+3] = 'agent_exploit'
            views_list.append(v)
    for x in range(basis_vfr-4): # set with 5 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            v[x+3] = 'agent_explore'
            v[x+4] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-4): # set with 5 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            v[x+3] = 'agent_exploit'
            v[x+4] = 'agent_exploit'
            views_list.append(v)
    for x in range(basis_vfr-5): # set with 6 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            v[x+3] = 'agent_explore'
            v[x+4] = 'agent_explore'
            v[x+5] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-5): # set with 6 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            v[x+3] = 'agent_exploit'
            v[x+4] = 'agent_exploit'
            v[x+5] = 'agent_exploit'
            views_list.append(v)
    for x in range(basis_vfr-6): # set with 7 explorer x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_explore'
            v[x+1] = 'agent_explore'
            v[x+2] = 'agent_explore'
            v[x+3] = 'agent_explore'
            v[x+4] = 'agent_explore'
            v[x+5] = 'agent_explore'
            v[x+6] = 'agent_explore'
            views_list.append(v)
    for x in range(basis_vfr-6): # set with 7 exploiter x each visual perturb
        for view in views:
            v = view.copy()
            v[x] = 'agent_exploit'
            v[x+1] = 'agent_exploit'
            v[x+2] = 'agent_exploit'
            v[x+3] = 'agent_exploit'
            v[x+4] = 'agent_exploit'
            v[x+5] = 'agent_exploit'
            v[x+6] = 'agent_exploit'
            views_list.append(v)
    for view in views:
        v = view.copy()
        v = ['agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore','agent_explore']
        views_list.append(v)
    for view in views:
        v = view.copy()
        v = ['agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit','agent_exploit']
        views_list.append(v)

    views = views_list
    num_conditions = int(len(views)/num_views)

    # iterate over runs
    data_all = []
    for name in names:
        gen, valfit = find_top_val_gen(name, 'cen')
        with open(fr'{data_dir}/{name}/{gen}_NNcen_pickle.bin','rb') as f:
            pv = pickle.load(f)
        envconf = de.dotenv_values(fr'{data_dir}/{name}/.env')
        NN, arch = reconstruct_NN(envconf, pv)

        acts = [agent_action_from_view(envconf, NN, v) for v in views] # iters over spatial/social views (only works for single vfr)
        act_array = np.zeros([num_views, num_conditions])

        x = 0
        borders = []
        for i in range(num_views):
            if data_type == 'act-abs':          act_array[i,0] = np.abs(acts[x])
            elif data_type == 'act-diff-abs':   act_array[i,0] = acts[x]
            x+=1
        last_loop = 1
        borders.append(last_loop)

        for a in range(8):
            for j in range(basis_vfr - a):
                for i in range(num_views):
                    if data_type == 'act-abs':          act_array[i,last_loop+j] = np.abs(acts[x])
                    elif data_type == 'act-diff-abs':   act_array[i,last_loop+j] = np.abs(acts[x] - act_array[i,0])
                    x+=1
            last_loop = last_loop+j+1
            borders.append(last_loop)
            for j in range(basis_vfr - a):
                for i in range(num_views):
                    if data_type == 'act-abs':          act_array[i,last_loop+j] = np.abs(acts[x])
                    elif data_type == 'act-diff-abs':   act_array[i,last_loop+j] = np.abs(acts[x] - act_array[i,0])
                    x+=1
            last_loop = last_loop+j+1
            borders.append(last_loop)
        data_all.append(act_array.flatten())
    param_list = np.array(data_all)
    return param_list


def count_followers(group_name, name_list):
    # param_list = indivperturb_data(name_list, metric='median')
    param_list = indivperturb_data(name_list, metric='num_found', types='nosocial')
    followers = 0
    for i, (OG, nosocial) in enumerate(param_list):
        # print(f'{i}: {OG, nosocial, nosocial-OG}')
        # if nosocial == 1000 and OG < 1000:
        if nosocial-OG < -1500:
        #     print('follower')
            followers += 1
    print(f'{group_name}: {followers} followers')
    return followers


def count_numfound(group_name, name_list):
    # param_list = indivperturb_data(name_list, metric='median')
    param_list = indivperturb_data(name_list, metric='num_found', types='OG')
    successes = 0
    for i, OG in enumerate(param_list):
        # print(f'{i}: {OG}')
        if OG > 1000:
        #     print('follower')
            successes += 1
    # print(f'{group_name}: {successes} successes')
    return successes



if __name__ == "__main__":

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/clustering'

    # group = 'main_fig'
    # # group = 'all'
    # IS, ISDP, DP, DPBD, BD, BDIS = gamut_dict_sort(group)
    # name_list = IS + ISDP + DP + DPBD + BD + BDIS
    # # name_list = IS + BDIS + BD
    # # name_list = IS + BD
    # class_types = np.array(len(IS)*[0] + len(ISDP)*[1] + len(DP)*[2] + len(DPBD)*[3] + len(BD)*[4] + len(BDIS)*[5])
    # # class_types = np.array(len(IS)*[0] + len(BD)*[1] + len(BDIS)*[2])
    # # class_types = np.array(len(IS)*[0] + len(BD)*[1])
    # # class_borders = [len(IS), len(IS)+len(ISDP), len(IS)+len(ISDP)+len(DP), len(IS)+len(ISDP)+len(DP)+len(DPBD), len(IS)+len(ISDP)+len(DP)+len(DPBD)+len(BD)]
    # labels = np.arange(0,len(name_list))

    # # param_list = extract_parameters(name_list)
    # param_list = extract_avgperfviews(name_list, space_step=5, orient_step=np.pi/256, data_type='spatial')
    # similarity_metrics = calculate_similarity_metrics(param_list, labels)

    # sim_fig = visualize_similarity(similarity_metrics, class_types, annote=False, tsne_perplexity=10, cluster_num=2, umap_nbs=10)
    # sim_fig.savefig("BD-BDIS-IS-visALL-dist8_action_similarity.png")
    # # sim_fig = visualize_similarity(similarity_metrics, class_types, annote=False, tsne_perplexity=10, cluster_num=3, umap_nbs=10)
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs_similarity.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-distALL_action-abs_similarity.png")

    # sim_fig = visualize_similarity(similarity_metrics, class_types, annote=False, clustering=True, cluster_num=2, tsne_perplexity=10, umap_nbs=10)
    # sim_fig.savefig("BD-BDIS-IS-visALL-dist8_action_similarity-clustering.png")
    # sim_fig = visualize_similarity(similarity_metrics, class_types, annote=False, clustering=True, cluster_num=3, tsne_perplexity=10, umap_nbs=10)
    # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs_similarity-clustering.png")
    # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-distALL_action-abs_similarity-clustering.png")

    # sim_fig = visualize_similarity(similarity_metrics, data_label=('decorr_time',name_list), annote=False, tsne_perplexity=10, umap_nbs=10)
    # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs_similarity-decorr_time.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-distALL_action-abs_similarity-decorr_time.png")
    # sim_fig = visualize_similarity(similarity_metrics, data_label=('de_mean',name_list), annote=False, tsne_perplexity=10, umap_nbs=10)
    # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs_similarity-de_mean.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-distALL_action-abs_similarity-de_mean.png")

    # sim_fig = visualize_similarity_tsne_perp(similarity_metrics, class_types)
    # # sim_fig = visualize_similarity_tsne_perp(similarity_metrics, data_label=('decorr_time',name_list))
    # # sim_fig = visualize_similarity_tsne_perp(similarity_metrics, data_label=('de_mean',name_list))
    # # sim_fig.savefig("BD-IS-visALL-dist8_action-abs-classtypes-perpALL-tSNE.png")
    # sim_fig.savefig("BD-BDIS-IS-visALL-dist8_action-abs-classtypes-perpALL-tSNE.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs-classtypes-perpALL-tSNE.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-distALL_action-abs-classtypes-perpALL-tSNE.png")

    # sim_fig = visualize_similarity_umap_perp(similarity_metrics, class_types)
    # # sim_fig = visualize_similarity_umap_perp(similarity_metrics, data_label=('decorr_time',name_list))
    # # sim_fig = visualize_similarity_umap_perp(similarity_metrics, data_label=('de_mean',name_list))
    # # sim_fig.savefig("BD-IS-visALL-dist8_action-abs-classtypes-perpALL-UMAP.png")
    # sim_fig.savefig("BD-BDIS-IS-visALL-dist8_action-abs-classtypes-perpALL-UMAP.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs-classtypes-perpALL-UMAP.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-distALL_action-abs-classtypes-perpALL-UMAP.png")

    # sim_fig = visualize_similarity_pacmap_perp(similarity_metrics, class_types, init='pca', mn=2, fp=2)
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs-classtypes-perpALL-PaCMAP.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs-classtypes-perpALL-pcaPaCMAP.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs-classtypes-perpALL-pcaPaCMAP-fp1.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs-classtypes-perpALL-pcaPaCMAP-fpp5.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs-classtypes-perpALL-pcaPaCMAP-mn1.png")
    # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs-classtypes-perpALL-pcaPaCMAP-mn2.png")

    # sim_fig = visualize_similarity_tables(similarity_metrics, class_borders)
    # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_param_similarity-tables.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action_similarity-tables.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-dist8_action-abs_similarity-tables.png")
    # # sim_fig.savefig("BD-BDIS-IS-ISDP-DP-DPBD-visALL-distALL_action-abs_similarity-tables.png")


    # n = 20
    # name_list = []
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW1_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW2_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW3_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW4_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW5_ND0_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # labels = np.arange(0,len(name_list))

    # name_list = []
    # n = 40
    # for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # n = 20
    # for name in [f'sc_N11_NRW0_ND10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # for name in [f'sc_N21_NRW0_ND20_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     name_list.append(name)
    # labels = np.arange(0,len(name_list))


    num_found_all = []
    num_followers_all = []

    name_list = []
    for x in range(20):
        name_list.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}')
    for x in range(20):
        name_list.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed10k_rep{x}')
    num_found_all.append(count_numfound('ND0', name_list))
    name_list = []
    n = 40
    for name in [f'sc_N2_NRW0_ND1_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        name_list.append(name)
    count_numfound('ND1', name_list)
    count_followers('ND1', name_list)
    name_list = []
    for name in [f'sc_N3_NRW0_ND2_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        name_list.append(name)
    count_numfound('ND2', name_list)
    count_followers('ND2', name_list)
    name_list = []
    for name in [f'sc_N4_NRW0_ND3_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        name_list.append(name)
    count_numfound('ND3', name_list)
    count_followers('ND3', name_list)
    name_list = []
    for name in [f'sc_N5_NRW0_ND4_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        name_list.append(name)
    count_numfound('ND4', name_list)
    count_followers('ND4', name_list)
    name_list = []
    for name in [f'sc_N6_NRW0_ND5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        name_list.append(name)
    count_numfound('ND5', name_list)
    count_followers('ND5', name_list)
    name_list = []
    n = 20
    for name in [f'sc_N11_NRW0_ND10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        name_list.append(name)
    count_numfound('ND10', name_list)
    count_followers('ND10', name_list)
    name_list = []
    for name in [f'sc_N21_NRW0_ND20_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        name_list.append(name)
    count_numfound('ND20', name_list)
    count_followers('ND20', name_list)


    # param_list = extract_parameters(name_list)

    # param_list = indivperturb_data(name_list)
    # for i, (OG,nosocial) in enumerate(param_list):
    #     print(f'{i}: {OG, nosocial, nosocial-OG}')

    # labels_perturb = []
    # class_types = []
    # for i, perturbs in enumerate(param_list):
    #     # og, allRW, allD, nosocial, selfsocial, ghostexplorer, ghostexploiter, N2ghostexplorer, N2ghostexploiter = perturbs
    #     og, allRW, allD, nosocial = perturbs
    #     label = ''
    #     if int(allD-og) > 400 and int(nosocial-og) > 300:
    #         label = f'{i}: IN/F'
    #         class_types.append(5)
    #     elif int(allD-og) > 400:
    #         label = f'{i}: IN'
    #         class_types.append(4)
    #     elif int(nosocial-og) > 300:
    #         label = f'{i}: F'
    #         class_types.append(0)
    #     elif int(allRW-og) >= 50 and int(allD-og) < 50 and int(nosocial-og) <= 200:
    #         label = f'{i}: F-mix'
    #         class_types.append(1)
    #     elif int(allRW-og) < 50 and int(allD-og) <= 400 and int(nosocial-og) < 50:
    #         label = f'{i}: IN-mix'
    #         class_types.append(3)
    #     else:
    #         label = f'{i}: bw'
    #         class_types.append(2)
    #     labels_perturb.append(label)
    #     # print(f'{name_list[i]}: {label}: {int(allRW-og)} | {int(allD-og)} | {int(nosocial-og)}')
    # class_types = np.array(class_types)

    # param_list = extract_avgperfviews(name_list, space_step=5, orient_step=np.pi/256, data_type='spatial')
    # param_list = extract_avgperfviews(name_list, space_step=5, orient_step=np.pi/256, data_type='spatial+social')
    # param_list = extract_avgperfviews(name_list, space_step=5, orient_step=np.pi/256, data_type='spatial+social x views')
    # param_list = extract_avgperfviews(name_list, space_step=5, orient_step=np.pi/256, data_type='social')
    # param_list = extract_avgperfviews(name_list, space_step=5, orient_step=np.pi/256, data_type='social-exploit')
    # param_list = extract_avgperfviews(name_list, space_step=5, orient_step=np.pi/256, data_type='social-explore')

    # # param_list = extract_acts(name_list, data_type='act-abs')
    # param_list = extract_acts(name_list, data_type='act-diff-abs')
    # print(param_list.shape)


    # similarity_metrics = calculate_similarity_metrics(param_list, labels)
    # # similarity_metrics = calculate_similarity_metrics(param_list, labels_perturb)

    # # sim_fig = visualize_similarity(similarity_metrics, tsne_perplexity=20, umap_nbs=100)
    # # # sim_fig.savefig("socN6_params_similarity.png")
    # # # sim_fig.savefig("socN6_action-abs_similarity.png")
    # # sim_fig.savefig(fr"{data_dir}/socN6_action-abs-1v_similarity.png")
    # # # sim_fig.savefig("socN6_action-abs-4v_similarity.png")
    # # # sim_fig.savefig("socN6_perturbs_similarity.png")

    # # sim_fig = visualize_similarity(similarity_metrics, class_types, cluster_num=6, tsne_perplexity=5, umap_nbs=10, pacmap_params=(15,.5,2), clustering=True)
    # # sim_fig.savefig(fr"{data_dir}/socN6_action-abs_similarity-classtypes.png")
    # # sim_fig = visualize_similarity(similarity_metrics, class_types, cluster_num=6, tsne_perplexity=20, umap_nbs=20, pacmap_params=(5,.5,2), clustering=True)
    # # sim_fig.savefig(fr"{data_dir}/socN6_action-diffabs_similarity-classtypes.png")
    # # sim_fig.savefig("socN6_perturbs_similarity-classtypes.png")

    # # sim_fig.savefig("socN6_action-abs-spatial_similarity.png")
    # # sim_fig.savefig("socN6_action-abs-social_similarity.png")
    # # sim_fig.savefig("socN6_action-abs-exploit_similarity.png")
    # # sim_fig.savefig("socN6_action-abs-explore_similarity.png")

    # # sim_fig.savefig("socN6_action-abs-labels_similarity.png")
    # # sim_fig.savefig("socN6_perturbs4-labels_similarity.png")

    # sim_fig = visualize_similarity_umap_perp(similarity_metrics)
    # # sim_fig.savefig("socN6_params_similarity-umap.png")
    # # sim_fig.savefig("socN6_action-abs_similarity-umap.png")
    # # sim_fig.savefig(fr"{data_dir}/socN6_action-abs-Xv_similarity-umap.png")
    # # sim_fig.savefig(fr"{data_dir}/socN6_action-diffabs-Xv_similarity-umap.png")
    # sim_fig.savefig(fr"{data_dir}/socNALL_action-diffabs-Xv_similarity-umap.png")
    # # sim_fig.savefig("socN6_perturbs_similarity-umap.png")

    # sim_fig = visualize_similarity_tsne_perp(similarity_metrics)
    # # sim_fig.savefig("socN6_params_similarity-tsne.png")
    # # sim_fig.savefig(fr"{data_dir}/socN6_action-abs-Xv_similarity-tsne.png")
    # # sim_fig.savefig(fr"{data_dir}/socN6_action-diffabs-Xv_similarity-tsne.png")
    # sim_fig.savefig(fr"{data_dir}/socNALL_action-diffabs-Xv_similarity-tsne.png")
    # # sim_fig.savefig("socN6_perturbs_similarity-tsne.png")

    # sim_fig = visualize_similarity_pacmap_perp(similarity_metrics, init='pca', mn=.5, fp=2)
    # # # sim_fig.savefig(fr"{data_dir}/socN6_action-abs-Xv_similarity-PaCMAP.png")
    # # sim_fig.savefig(fr"{data_dir}/socN6_action-diffabs-Xv_similarity-PaCMAP.png")
    # sim_fig.savefig(fr"{data_dir}/socNALL_action-diffabs-Xv_similarity-PaCMAP.png")
    # # for mn in [.5,1,2]:
    # #     for fp in [.5,1,2]:
    # #         sim_fig = visualize_similarity_pacmap_perp(similarity_metrics, init='pca', mn=mn, fp=fp)
    # #         # sim_fig.savefig(fr"{data_dir}/socN6_action-abs-Xv_similarity-PaCMAP.png")
    # #         sim_fig.savefig(fr"{data_dir}/socN6_action-diffabs-Xv_similarity-PaCMAP_mn{mn}_fp{fp}.png")

    # plt.show()