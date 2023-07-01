import kaggle
import os

# this will install dataset to your local machine
# this dataset is not too big so it is safe to install it

# later, we will add variuous datasets here
dataset_id = 'akhileshdkapse/super-image-resolution'
output_dir = 'datasets'

os.makedirs(output_dir, exist_ok=True)

kaggle.api.dataset_download_files(dataset_id, path=output_dir, unzip=True)
