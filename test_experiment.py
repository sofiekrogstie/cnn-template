"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 100 first images from prediction set.
"""

from deoxys.experiment import Experiment
# from deoxys.utils import read_file
import argparse
import os
# from pathlib import Path
# from comet_ml import Experiment as CometEx
import tensorflow as tf
import customize_obj

if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--model_checkpoint_period", default=25, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=25, type=int)

    args = parser.parse_args()

    # 42 images for 2d, 15 images for 3d
    img_num = 100

    if '3d' in args.log_folder:
        img_num = 40

    # config = read_file(args.config_file)
    if os.path.isfile(args.config_file) and args.config_file.endswith('h5'):
        initial_epoch = int(args.config_file[-6:-3])
    else:
        raise RuntimeError('Not a model file')
    (
        Experiment(log_base_path=args.log_folder)
        .from_file(args.config_file)
        .run_test(masked_images=[i for i in range(img_num)])
    )
