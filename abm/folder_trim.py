from pathlib import Path
import os, shutil


root_dir = Path(__file__).parent / fr'data/simulation_data'
for file in os.listdir(root_dir):

    dir = Path(root_dir, file)
    print(dir)

    for g in range(1000):

        if os.path.isdir(Path(dir,fr'gen{g}')):

            for file in os.listdir(Path(dir,fr'gen{g}')):

                if file.startswith('NN0'):
                
                    if not file.endswith('png'):

                        shutil.move(fr'{dir}/gen{g}/{file}/NN_pickle.bin', fr'{dir}/gen{g}_NN0.bin')

            shutil.rmtree(fr'{dir}/gen{g}')

        # else: 
        #     print(f'failed for: {dir,g}')
