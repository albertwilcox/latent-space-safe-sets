
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')

import latentsafesets.utils as utils
from latentsafesets.utils.arg_parser import parse_args

import os
from PIL import Image
from tqdm import tqdm


def main():
    params = parse_args()
    env_name = params['env']#spb
    frame_stack = params['frame_stack']#1 or not? In spb is is not stacking
    demo_trajectories = []
    for count, data_dir in list(zip(params['data_counts'], params['data_dirs'])):#see 180-200
        demo_trajectories += utils.load_trajectories(count, file='data/' + data_dir)#98
        #count=50, each for non-constraint and constraint, thus together 100

    i = 0
    save_folder = os.path.join('data_images', env_name)#data_images/spb
    os.makedirs(save_folder, exist_ok=True)#make the directory of data_images/spb or /push!
    for trajectory in tqdm(demo_trajectories):#show progress bar! 100 trajectories!
        for frame in trajectory:#100 frames in one trajectory!
            if frame_stack == 1:
                im = Image.fromarray(frame['obs'].transpose((1, 2, 0)))#maybe channel last
                im.save(os.path.join(save_folder, '%d.png' % i))#save 3 channel obs image png
            else:
                for j in range(frame_stack):
                    im = Image.fromarray(frame['obs'][j].transpose((1, 2, 0)))
                    im.save(os.path.join(save_folder, '%d_%d.png' % (i, j)))
            i += 1


if __name__ == '__main__':
    main()
