# script for doing runs with multiple simulations/parameters
# run via `python path_to_exp_file.py`

from abm.metarunner.metarunner import Tunable, Constant, MetaProtocol, TunedPairRestrain
import numpy as np
import os
EXP_NAME = os.getenv("EXPERIMENT_NAME", "small_patch_only")
if EXP_NAME == "":
    raise Exception("No experiment name has been passed")

description_text = f"""
Experiment file using the MetaRunner interfacing language to define a set of criteria for batch simulations

Title:      Experiment : {EXP_NAME}
Date:       30.05.2022
Goal:       Check locations of N=10, N_RESOURCES=1, RADIUS_RESOURCE=1 as test.

Defined by: nb
"""
# parameters can kept mostly untouched
# Defining fixed criteria for all automized simulations/experiments
arena_w = 500
arena_h = 500
fixed_criteria = [
    Constant("USE_IFDB_LOGGING", 0),
    Constant("USE_RAM_LOGGING", 1),
    Constant("USE_ZARR_FORMAT", 1),
    Constant("SAVE_CSV_FILES", 1),
    Constant("WITH_VISUALIZATION", 0),  # how does the simulation speed scale with N
    Constant("TELEPORT_TO_MIDDLE", 0),
    Constant("GHOST_WHILE_EXPLOIT", 1),
    Constant("PATCHWISE_SOCIAL_EXCLUSION", 1),
    Constant("POOLING_TIME", 0),
    Constant("MOV_EXP_VEL_MIN", 1),
    Constant("MOV_EXP_VEL_MAX", 1),
    Constant("MOV_REL_DES_VEL", 1),
    Constant("SHOW_VISUAL_FIELDS", 0),
    Constant("SHOW_VISUAL_FIELDS_RETURN", 0),
    Constant("SHOW_VISION_RANGE", 0),
    Constant("ENV_WIDTH", arena_w),
    Constant("ENV_HEIGHT", arena_h),
    Constant("VISUAL_FIELD_RESOLUTION", 1200),
    Constant("VISION_RANGE", 2000),
    Constant("AGENT_CONSUMPTION", 1),
    Constant("RADIUS_AGENT", 10),
    Constant("MAX_RESOURCE_QUALITY", -1),  # so that the minimum value will be used as definite
    Constant("MAX_RESOURCE_PER_PATCH", -1),  # so that the minimum value will be used as definite
    Constant("MOV_EXP_TH_MIN", -0.25),
    Constant("MOV_EXP_TH_MAX", 0.25),
    Constant("MOV_REL_TH_MAX", 0.5),
    Constant("CONS_STOP_RATIO", 0.1),
    Constant("REGENERATE_PATCHES", 1),
    Constant("PATCH_BORDER_OVERLAP", 1),     
    Constant("DEC_FN", 0.5),
    Constant("DEC_FR", 0.5),
    Constant("DEC_TAU", 10),
    Constant("DEC_BW", 0),
    Constant("DEC_WMAX", 1),
    Constant("DEC_BU", 0),
    Constant("DEC_UMAX", 1),
    Constant("DEC_GW", 0.085),
    Constant("DEC_GU", 0.085),
    Constant("DEC_TW", 0.5),
    Constant("DEC_TU", 0.5)
]

# Defining decision param
sum_resources = 3000
arena_size = arena_w * arena_h
# keeping the covered area on 20% on overall area
overall_res_area = int(arena_size * 0.2)

criteria_exp = [
    Constant("N", 10),
    Constant("VISUAL_EXCLUSION", 0),
    Constant("AGENT_FOV", 1),  # unlimited
    Constant("DEC_EPSW", 0.25), # social excitability
    Constant("DEC_EPSU", 1),
    Constant("MIN_RESOURCE_QUALITY", 0.25),
    Constant("MIN_RESOURCE_PER_PATCH", 30),
    Constant("DEC_SWU", 0),
    Constant("DEC_SUW", 0),
    Constant("RADIUS_RESOURCE", 1),
    Constant("N_RESOURCES", 1),
    Constant("T", 1)
]

# restrain pair of tuned parameters
# one can also change number of batches num_batches

# Creating metaprotocol and add defined criteria
mp = MetaProtocol(experiment_name=EXP_NAME, num_batches=10, parallel=True,
                  description=description_text, headless=True)
for crit in fixed_criteria:
    mp.add_criterion(crit)
for crit in criteria_exp:
    mp.add_criterion(crit)

# Generating temporary env files with criterion combinations. Comment this out if you want to continue simulating due
# to interruption
mp.generate_temp_env_files()

# Running the simulations
mp.run_protocols()
