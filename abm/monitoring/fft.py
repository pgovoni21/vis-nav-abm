from abm.monitoring.trajs import find_top_val_gen

from pathlib import Path
import dotenv as de
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy import fft

def calculate_2dft(input):
    ft = input
    ft = fft.ifftshift(input)
    ft = fft.fft2(ft)
    ft = fft.fftshift(ft)
    return ft

def calculate_inv_2dft(input):
    ft = input
    ft = abs(ft)
    ft = fft.ifft2(ft)
    return ft

def build_circle(grid_size, center_pos, radius):

    grid = np.zeros(grid_size)

    x_c, y_c = center_pos

    # Create index arrays to z
    I,J = np.meshgrid(np.arange(grid.shape[0]),
                    np.arange(grid.shape[1]))

    # calculate distance of all points to centre
    dist = np.sqrt((I - x_c)**2 + (J - y_c)**2)

    # Assign value of 1 to those points where dist<cr:
    grid[np.where(dist < radius)] = 1

    return grid


def plot_hist_FFT(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True, extra=''):
    # print(f'plotting {exp_name}, {gen_ext}, {space_step}, {int(np.pi/orient_step)}, {timesteps}, {rank}, {int(eye)}, {extra}, {landmarks}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)
    x_max,y_max = tuple(eval(envconf['ENV_SIZE']))

    if extra == '':
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    else:
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_{extra}'

    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    
    # only xy coords + after first 25 timesteps + before 250 timesteps + flatten all trajs
    ag_data = ag_data[:,25:250,:2]
    ag_data = ag_data.reshape(ag_data.shape[0]*ag_data.shape[1],2)
    # print('ag_data shape', ag_data.shape)
    x = ag_data[:,0]
    y = ag_data[:,1]


    # for bs in [250,500,1000]:
    for bs in [500]:
        for r_filter in [0]:
        # for r_filter in [0,1,2,5,10,25,50,100]:
            for r_blank in [50,100,150]:

                bins = np.linspace(0, x_max, bs+1)
                # print('bins', bins)

                hist,_,_ = np.histogram2d(x, y, bins=bins)
                hist = hist.transpose() # flip axes for plotting
                # print('hist shape', hist.shape)


                # # cancel out patch

                # center_pos = (hist.shape[0]*.4, hist.shape[1]*.6)
                # scal_fac = hist.shape[0]/1000
                # mask = build_circle(hist.shape, center_pos, radius=r_blank * scal_fac)
                # # hist = np.where(mask == 1, hist, 0) #lo
                # hist = np.where(mask == 0, hist, 0) # blank out inner circle

                # hist

                fig, axs = plt.subplots(1,3, figsize=(15, 8))

                ax = axs[0]
                map = ax.imshow(hist, origin='lower', norm=LogNorm())
                ax.set_title('hist - blank patch - log')
                ax.set_xlim(0, hist.shape[0])
                ax.set_ylim(0, hist.shape[1])
                fig.colorbar(map, ax=ax, fraction=0.046, pad=0.04)


                # FFT

                ft = fft.fft2(hist)
                ft = fft.fftshift(ft)

                if r_filter == 0:
                    pass
                else:
                    center_pos = (hist.shape[0]/2, hist.shape[1]/2)
                    mask = build_circle(hist.shape, center_pos, radius=r_filter)
                    # ft = np.where(mask == 1, ft, 0) #lo
                    ft = np.where(mask == 0, ft, 0) #hi

                ax = axs[1]
                map = ax.imshow(abs(ft), origin='lower', norm=LogNorm())
                if r_filter == 0:
                    ax.set_title('fwd FT - fftshift - abs - log')
                else:
                    ax.set_title(f'fwd FT - fftshift - hipass (r{r_filter}) - abs - log')
                ax.set_xlim(0, hist.shape[0])
                ax.set_ylim(0, hist.shape[1])
                fig.colorbar(map, ax=ax, fraction=0.046, pad=0.04)


                # autocorrelation

                ft = ft * np.conj(ft)
                # ft = ft * ft
                ft = fft.ifft2(ft)
                ft = fft.fftshift(ft)

                ax = axs[2]
                map = ax.imshow(abs(ft), origin='lower', norm=LogNorm())
                if r_filter == 0:
                    ax.set_title('fwd FT - x*conj(x) - inv FT - fftshift - abs - log')
                else:
                    ax.set_title(f'fwd FT - hipass (r{r_filter}) - x*conj(x) - inv FT - fftshift - abs - log')
                    # ax.set_title(f'fwd FT - hipass (r{r}) - x^2 - inv FT - fftshift - abs - log')
                ax.set_xlim(0, hist.shape[0])
                ax.set_ylim(0, hist.shape[1])
                fig.colorbar(map, ax=ax, fraction=0.046, pad=0.04)

                plt.tight_layout()
                if r_filter == 0:
                    # plt.savefig(fr'{save_name}_fft_{str(bs)}_log_conj_blankpatch_r{str(r_blank)}.png')
                    plt.savefig(fr'{save_name}_fft_{str(bs)}_log_conj.png')
                else:
                    plt.savefig(fr'{save_name}_fft_{str(bs)}_log_hipass_r{str(r_filter)}.png')
                plt.close()




def plot_hist_FFT_hipass(exp_name, gen_ext, space_step, orient_step, timesteps, rank='cen', eye=True, extra=''):

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    env_path = fr'{data_dir}/{exp_name}/.env'
    envconf = de.dotenv_values(env_path)
    x_max,y_max = tuple(eval(envconf['ENV_SIZE']))

    if extra == '':
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}'
    else:
        save_name = fr'{data_dir}/traj_matrices/{exp_name}_{gen_ext}_c{space_step}_o{int(np.pi/orient_step)}_t{timesteps}_{rank}_e{int(eye)}_{extra}'

    with open(save_name+'.bin', 'rb') as f:
        ag_data = pickle.load(f)
    
    # only xy coords + after first 25 timesteps + before 250 timesteps + flatten all trajs
    ag_data = ag_data[:,25:250,:2]
    ag_data = ag_data.reshape(ag_data.shape[0]*ag_data.shape[1],2)
    print('ag_data shape', ag_data.shape)
    x = ag_data[:,0]
    y = ag_data[:,1]



    bs = 1000
    for bs in [500, 1000, 2000]:
        bins = np.linspace(0, x_max, bs+1)

        radii = [1,2,5,10,100,200]

        hist,_,_ = np.histogram2d(x, y, bins=bins)
        hist = hist.transpose() # flip axes for plotting

        fig, ax = plt.subplots(2,len(radii), figsize=(25, 10))

        ft = fft.fft2(hist)

        for i, r in enumerate(radii):

            center_pos = (hist.shape[0]/2, hist.shape[0]/2)
            mask = build_circle(hist.shape, center_pos, radius=r)
            pass_ft = np.where(mask == 0, ft, 0) #hi
            inv_ft = fft.ifft2(pass_ft)

            ax[0,i].imshow(np.log(abs(inv_ft)), origin='lower')
            ax[0,i].set_title(f'hi, r {r}')

            pass_ft = np.where(mask == 1, ft, 0) #lo
            inv_ft = fft.ifft2(pass_ft)

            ax[1,i].imshow(np.log(abs(inv_ft)), origin='lower')
            ax[1,i].set_title(f'lo, r {r}')

        plt.tight_layout()
        plt.savefig(fr'{save_name}_fft_bs{str(bs)}_hilopass.png')
        plt.close()




if __name__ == '__main__':


    ## traj / Nact
    space_step = 25
    orient_step = np.pi/8

    timesteps = 500

    names = []

    # names.append('sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep18')
    for i in [1,3,4,10]:
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{str(i)}')
    
    for i in [3,5]:
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{str(i)}')
    for i in [3]:
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{str(i)}')
    for i in [5]:
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep{str(i)}')
    for i in [1]:
        names.append(f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n4_rep{str(i)}')
    
    names.append('sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_angl_n10_rep9')
    names.append('sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmdistpost_n100_rep17')
    names.append('sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmangle_n10_rep16')
    names.append('sc_lm_CNN14_FNN2_p50e20_vis12_lm100_lmradius_n50_rep15')

    for name in names:
        gen, valfit = find_top_val_gen(name, 'cen')
        print(f'running: {name} @ {gen} w {valfit} fitness')

        plot_hist_FFT(name, gen, space_step, orient_step, timesteps)
        # plot_hist_FFT(name, gen, space_step, orient_step, timesteps, extra='n0')
        # plot_hist_FFT_hipass(name, gen, space_step, orient_step, timesteps)