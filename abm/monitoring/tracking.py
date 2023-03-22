"""
@author: mezdahun
@description: Helper functions for InfluxDB
"""
# import importlib
# import abm.contrib.tracking_params as tp

from pathlib import Path
import zarr

resources_dict = {}
agents_dict = {}


def mode_to_int(mode):
    """converts a string agent mode flag into an int"""
    if mode == "explore":
        return int(0)
    elif mode == "exploit":
        return int(1)
    elif mode == "collide":
        return int(2)


def save_agent_data_RAM(sim):
    """Saving relevant agent data
    if multiple simulations are running in parallel a uuid hash must be passed as experiment hash to find
    the unique measurement in the database
    """
    global agents_dict
    # if sim.t % 500 == 0:
    #     print(f"Agent data size in memory: {sys.getsizeof(agents_dict)/1024} MB", )
    for ag in sim.agents:
        if ag.id not in list(agents_dict.keys()): # setup subdict
            agents_dict[ag.id] = {}
            agents_dict[ag.id][f"pos_x"] = []
            agents_dict[ag.id][f"pos_y"] = []
            agents_dict[ag.id][f"mode"] = []
            agents_dict[ag.id][f"collected_r"] = []

        # convert positional coordinates
        x,y = ag.pt_center
        pos_x = x - sim.window_pad
        pos_y = sim.y_max - y

        # input data of current time step
        agents_dict[ag.id][f"pos_x"].append(pos_x)
        agents_dict[ag.id][f"pos_y"].append(pos_y)
        agents_dict[ag.id][f"mode"].append(mode_to_int(ag.mode))
        agents_dict[ag.id][f"collected_r"].append(ag.collected_r)


def save_resource_data_RAM(sim):
    """Saving relevant resource patch data 
    if multiple simulations are running in parallel a uuid hash must be passed as experiment hash to find
    the unique measurement in the database"""
    global resources_dict
    # if sim.t % 500 == 0:
    #     print(f"Resource data size in memory: {sys.getsizeof(resources_dict)/1024} MB", )
    for res in sim.resources:
        if res.id not in list(resources_dict.keys()): # setup subdict
            
            # convert positional coordinates
            x,y = res.pt_center
            pos_x = x - sim.window_pad
            pos_y = sim.y_max - y

            resources_dict[res.id] = {}
            resources_dict[res.id]["start_time"] = sim.t
            resources_dict[res.id]["pos_x"] = pos_x
            resources_dict[res.id]["pos_y"] = pos_y
            resources_dict[res.id]["radius"] = res.radius
            resources_dict[res.id]["resrc_left"] = []
            resources_dict[res.id]["quality"] = []

        # input data of current time step
        resources_dict[res.id]["resrc_left"].append(res.resrc_left)
        resources_dict[res.id]["quality"].append(res.quality)


def save_zarr_file(sim_time, save_ext, print_enabled=False):
    """Saving agent/resource dictionaries as zarr file
    if multiple simulations are running in parallel a uuid hash must be passed as experiment hash to find
    the unique measurement in the database
    
    Note - fwd slashes needed for Linux folder pathing, while Windows is ambivalent
    """
    global agents_dict, resources_dict

    ### construct save directory according to save_ext, timestamping if not provided
    root_dir = Path(__file__).parent.parent.parent
    if save_ext:
        save_dir = Path(root_dir, 'abm/data/simulation_data', save_ext)
        # print(f'Saving: {save_dir}')
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_dir = Path(root_dir, 'abm/data/simulation_data', timestamp)
        print(f'Saving: {save_dir}')

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if print_enabled: print(f"Saving in {save_dir}")

    ### construct agent N-dimensional array of saved data
    num_ag = len(agents_dict)
    num_ag_data_entries = len(list(agents_dict[0]))
    save_data_ag_shape = (num_ag, sim_time, num_ag_data_entries)
    if print_enabled: print(f"Saving agent data as {len(save_data_ag_shape)}-D zarr array of shape {save_data_ag_shape}")

    ag_zarr = zarr.open(Path(save_dir, "ag.zarr"), mode='w', shape=save_data_ag_shape,
                            chunks=save_data_ag_shape, dtype='float')

    # populate zarr array from dict
    for ag_id,_ in agents_dict.items():
        ag_zarr[ag_id, :, 0] = agents_dict[ag_id]['pos_x']
        ag_zarr[ag_id, :, 1] = agents_dict[ag_id]['pos_y']
        ag_zarr[ag_id, :, 2] = agents_dict[ag_id]['mode']
        ag_zarr[ag_id, :, 3] = agents_dict[ag_id]['collected_r']

    ### construct resource N-dimensional array of saved data
    num_res = len(resources_dict)
    num_res_data_entries = len(list(resources_dict[0]))
    save_data_res_shape = (num_res, sim_time, num_res_data_entries)
    if print_enabled: print(f"Saving resource data as {len(save_data_res_shape)}-D zarr array of shape {save_data_res_shape}")

    res_zarr = zarr.open(Path(save_dir, "res.zarr"), mode='w', shape=save_data_res_shape,
                            chunks=save_data_res_shape, dtype='float')

    # populate zarr array from dict
    for res_id,_ in resources_dict.items():

        start_time = resources_dict[res_id]['start_time']
        end_time = len(resources_dict[res_id]['resrc_left']) + start_time

        res_zarr[res_id, 0, 0] = resources_dict[res_id]['pos_x']
        res_zarr[res_id, 0, 1] = resources_dict[res_id]['pos_y']
        res_zarr[res_id, 0, 2] = resources_dict[res_id]['radius']
        res_zarr[res_id, start_time:end_time, 3] = resources_dict[res_id]['resrc_left']
        res_zarr[res_id, start_time:end_time, 4] = resources_dict[res_id]['quality']

    clean_global_dicts()

    return ag_zarr, res_zarr

def clean_global_dicts():
    # clean global data structures
    # - after loading instance info to file
    # - for crashed simulations (without offloading to a file)

    global agents_dict, resources_dict

    resources_dict = {}
    agents_dict = {}