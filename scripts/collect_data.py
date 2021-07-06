from latentsafesets.utils.arg_parser import parse_args
import latentsafesets.utils as utils
from latentsafesets.utils.teacher import ConstraintTeacher, ReacherTeacher,\
    ReacherConstraintTeacher, StrangeTeacher, PushTeacher, OutburstPushTeacher, \
    SimplePointBotTeacher
import latentsafesets.utils.plot_utils as pu

import logging
import os
log = logging.getLogger("collect")


env_teachers = {
    'spb': [
        SimplePointBotTeacher, ConstraintTeacher, StrangeTeacher
    ],
    'reacher': [
        ReacherTeacher, ReacherConstraintTeacher, StrangeTeacher
    ],
    'push': [
        PushTeacher, OutburstPushTeacher
    ],
}


def generate_teacher_demo_data(env, data_dir, teacher, n=100, noisy=False, logdir=None):
    log.info("Generating teacher demo trajectories")
    file = os.path.join('data', data_dir)
    if not os.path.exists(file):
        os.makedirs(file)
    else:
        raise RuntimeError("Directory %s already exists." % file)
    teacher = teacher(env, noisy=noisy)
    demonstrations = []
    for i in range(n):
        traj = teacher.generate_trajectory()
        reward = sum([frame['reward'] for frame in traj])
        print('Trajectory %d, Reward %d' % (i, reward))
        demonstrations.append(traj)
        utils.save_trajectory(traj, file, i)
        if i < 50 and logdir is not None:
            pu.make_movie(traj, os.path.join(logdir, '%s_%d.gif' % (data_dir, i)))
    return demonstrations


def main():
    params = parse_args()

    logdir = utils.get_file_prefix()
    os.makedirs(logdir)
    utils.init_logging(logdir)

    env = utils.make_env(params)

    teachers = env_teachers[params['env']]
    data_dirs = params['data_dirs']
    data_counts = params['data_counts']

    for teacher, data_dir, count in list(zip(teachers, data_dirs, data_counts)):
        generate_teacher_demo_data(env, data_dir, teacher, count, True, logdir)


if __name__ == '__main__':
    main()
