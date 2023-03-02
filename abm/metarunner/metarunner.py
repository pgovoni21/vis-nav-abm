"""
metarunner.py: including the main classes and methods to programatically run metaprotocols of simulations, i.e.
     for some parameter search.
"""
import shutil
import numpy as np
import os
from pathlib import Path
import itertools
from dotenv import dotenv_values
import warnings
from abm import app
import glob
from time import sleep

class Constant:
    """A constant parameter value for a given parameter that shall be used for simulations"""

    def __init__(self, var_name, constant):
        """defines a single variable value with name"""
        self.tunable = Tunable(var_name, values_override=[constant])
        self.name = self.tunable.name

    def get_values(self):
        return self.tunable.values

    def print(self):
        """printing method"""
        print(f"---Added Constant:\t {self.tunable.name} = {self.tunable.values[0]}")

class TunedPairRestrain:
    """Parameter pair to be restrained together with multiplication"""
    def __init__(self, var_name1, var_name2, restrained_product):
        self.var1 = var_name1
        self.var2 = var_name2
        self.product_restrain = restrained_product

    def get_vars(self):
        return [self.var1, self.var2]

    def print(self):
        """printing method"""
        print(f"Product of {self.var1} and {self.var2} should be {self.product_restrain}")

class Tunable:
    """A parameter range in which we want to loop through (explore)"""

    def __init__(self, var_name, min_v=None, max_v=None, num_data_points=None, values_override=None):
        """init method of the Tunable class. We want to loop through a parameter defined with var_name from the
        min value to the max value with num_data_points number of individual parameter values between

        In case we have specific values to loop through we can pass a list of values instead of borders and number
        of datapoints."""

        if min_v is None and values_override is None:
            raise Exception("Neither value borders nor override values have been given to create Tunable!")
        elif min_v is not None and values_override is not None:
            warnings.warn("Both value borders and override values are defined when creating Tunable, using override"
                          "values as default!")

        self.name = var_name
        if values_override is None:
            self.min_val = min_v
            self.max_val = max_v
            self.n_data = num_data_points
            self.generated = True
            self.values = np.linspace(self.min_val, self.max_val, num=self.n_data, endpoint=True)
        else:
            self.min_val = min(values_override)
            self.max_val = max(values_override)
            self.n_data = len(values_override)
            self.generated = False
            self.values = values_override

    def print(self):
        """printing method"""
        print(f"---Added Tunable:\t {self.name} = {self.values}")

    def get_values(self):
        return self.values


class MetaProtocol:
    """Metaprotocol class that is initialized with Tunables and runs through the desired simulations accordingly"""

    def __init__(self, experiment_name, num_batches=1, parallel=False, description=None, headless=False):
        self.tunables = []
        self.tuned_pairs = []
        self.q_tuned_pairs = []
        self.experiment_name = experiment_name
        self.num_batches = num_batches
        self.description = description
        self.headless = headless
        self.parallel_run = parallel

        self.root_dir = Path(__file__).parent.parent.parent
        self.temp_dir = Path(self.root_dir, "abm/data/metaprotocol/temp", self.experiment_name)
        self.save_dir = Path(self.root_dir, "abm/data/simulation_data")

    def add_criterion(self, criterion):
        """Adding a criterion to the metaprotocol as a Tunable or Constant"""
        self.tunables.append(criterion)
        criterion.print()

    def add_tuned_pair(self, tuned_pair):
        self.tuned_pairs.append(tuned_pair)
        print("---Added new restrained pair: ")
        tuned_pair.print()

    def add_quadratic_tuned_pair(self, tuned_pair):
        self.q_tuned_pairs.append(tuned_pair)
        print("---Added new restrained *quadratic* pair: ")
        tuned_pair.print()

    def consider_tuned_pairs(self, combos):
        """removing combinations from a list of combinations where a tuned pair criterion is not met"""
        tunable_names = [t.name for t in self.tunables]
        new_combos = combos.copy()
        for tuned_pair in self.tuned_pairs:
            for i, combo in enumerate(combos):
                print("combo", combo)
                product = 1
                for j, value in enumerate(combo):
                    name = tunable_names[j]
                    if name in tuned_pair.get_vars():
                        product *= value
                if product != tuned_pair.product_restrain:
                    print("POP")
                    new_combos.remove(combo)
        for tuned_pair in self.q_tuned_pairs:
            for i, combo in enumerate(combos):
                print("combo", combo)
                product = 1
                for j, value in enumerate(combo):
                    name = tunable_names[j]
                    if name == tuned_pair.get_vars()[0]:
                        product *= value
                    elif name == tuned_pair.get_vars()[1]:
                        product *= value * value
                if not np.isclose(product,tuned_pair.product_restrain):
                    print("POP")
                    try:
                        new_combos.remove(combo)
                    except ValueError:
                        print("combo already removed")
        return new_combos

    def generate_temp_env_files(self):
        """generating a batch of env files that will describe the metaprotocol in a temporary folder"""

        # check for existing files in temp + save directories, overwrite if found
        if os.path.isdir(self.temp_dir):
            warnings.warn("Temporary directory for env files is not empty and will be overwritten")
            shutil.rmtree(self.temp_dir)

        if os.path.isdir(Path(self.save_dir, self.experiment_name)):
            warnings.warn("Save directory for env + zarr files is not empty and will be overwritten")
            shutil.rmtree(Path(self.save_dir, self.experiment_name))

        # generate combinations of variables
        tunable_names = [t.name for t in self.tunables]
        tunable_values = [t.get_values() for t in self.tunables]
        combos = list(itertools.product(*tunable_values))

        # remove generated combos that do not follow constraint
        combos = self.consider_tuned_pairs(combos)

        print(f"Generating {len(combos)} env files for simulations")

        # for number of metaprotocol batches
        for nb in range(self.num_batches):

            # for each combination of variables
            for i, combo in enumerate(combos):

                # initialize env dict
                new_envconf = dict()

                # for each contant/tunable variable
                for j, value in enumerate(combo):
                    name = tunable_names[j]
                    if not isinstance(value, bool):
                        new_envconf[name] = value
                    else:
                        new_envconf[name] = int(value)

                # create save folder + append sim_save_name to env file
                save_ext = Path(self.experiment_name, f"batch_{nb}", f"combo_{i+1}")
                os.makedirs(Path(self.save_dir, save_ext))
                new_envconf["SAVE_EXT"] = save_ext

                # generate temporary env file
                os.makedirs(self.temp_dir, exist_ok=True)
                file_path = Path(self.temp_dir, f"{self.experiment_name}_b{nb}_c{i+1}.env")
                with open(file_path, "a") as file:
                    for k, v in new_envconf.items():
                        file.write(f"{k}={v}\n")

        print(f"Env files generated according to criterions!")

    def save_description(self):
        """Saving description text as txt file in the experiment folder"""
        if self.description is not None:
            description_path = Path(self.save_dir, self.experiment_name, "README.txt")
            os.makedirs(self.save_dir, exist_ok=True)
            with open(description_path, "w") as readmefile:
                readmefile.write(self.description)

    def run_protocol(self, temp_env, project="Base"):
        """Runs a single simulation run according to an env file given by the env path"""

        # pull protocol name + set as env path
        protocol_name = Path(temp_env).stem
        os.environ["EXPERIMENT_NAME"] = protocol_name

        # call save extension (batch#/combo#)
        envconf = dotenv_values(temp_env)
        save_ext = envconf["SAVE_EXT"]

        # move temporary .env file to root_dir + save_dir
        shutil.copyfile(temp_env, f"{Path(self.root_dir, protocol_name)}.env") # concatenates with protocol stem
        shutil.copyfile(temp_env, Path(self.save_dir, save_ext, ".env")) # creates invidual env for the folder
        
        # run sim via current .env file (in root_dir)
        if project == "Base":
            app.start()
        # elif project == "CoopSignaling":
        #     from abm import app_collective_signaling
        #     app_collective_signaling.start(parallel=self.parallel_run, headless=self.headless)

        # remove temp + current .env files
        os.remove(temp_env)
        os.remove(f"{Path(self.root_dir, protocol_name)}.env")

        sleep(2)

    def run_protocols(self, project="Base"):
        """Iterates through list of protocols in temp env folder"""

        # save experiment README.txt
        self.save_description()
        
        # iterates through list
        for i, temp_env in enumerate(self.temp_dir.glob("*.env")):
            self.run_protocol(temp_env, project=project)

        # remove temp folder
        os.rmdir(self.temp_dir)