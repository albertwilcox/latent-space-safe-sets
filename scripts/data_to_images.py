import latentsafesets.utils as utils
from latentsafesets.utils.arg_parser import parse_args

import os
from PIL import Image
from tqdm import tqdm


def main():
    params = parse_args()
    env_name = params['env']
    frame_stack = params['frame_stack']
    demo_trajectories = []
    for count, data_dir in list(zip(params['data_counts'], params['data_dirs'])):
        demo_trajectories += utils.load_trajectories(count, file='data/' + data_dir)

    i = 0
    save_folder = os.path.join('data_images', env_name)
    os.makedirs(save_folder, exist_ok=True)
    for trajectory in tqdm(demo_trajectories):
        for frame in trajectory:
            if frame_stack == 1:
                im = Image.fromarray(frame['obs'].transpose((1, 2, 0)))
                im.save(os.path.join(save_folder, '%d.png' % i))
            else:
                for j in range(frame_stack):
                    im = Image.fromarray(frame['obs'][j].transpose((1, 2, 0)))
                    im.save(os.path.join(save_folder, '%d_%d.png' % (i, j)))
            i += 1


if __name__ == '__main__':
    main()
