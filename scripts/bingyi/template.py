import tyro
import os 
from tqdm.auto import tqdm
from os.path import join
import numpy as np

def rgba2rgb(
        input_path: str = '/mnt/bn/haotongdata/Datasets/nerf_synthetic/lego/train',
        output_path: str = '/mnt/bn/haotongdata/Datasets/nerf_synthetic/lego/train_rgb'
    ) -> None:
    """Parse metric from txt."""
    image_names = sorted(os.listdir(input_path))
    for image_name in tqdm(image_names):
        pass
    
if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "rgba2rgb": rgba2rgb,
        }
    )

