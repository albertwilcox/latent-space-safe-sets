
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')

from latentsafesets.utils.arg_parser import parse_args
import latentsafesets.utils as utils
from latentsafesets.utils.teacher import ConstraintTeacher, ReacherTeacher,\
    ReacherConstraintTeacher, StrangeTeacher, PushTeacher, OutburstPushTeacher, \
    SimplePointBotTeacher
import latentsafesets.utils.plot_utils as pu

import logging
import os
log = logging.getLogger("collect")


env_teachers = {#it is a dictionary, right?
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
    file = os.path.join('data', data_dir)#create the data folder!
    if not os.path.exists(file):#['SimplePointBot','SimplePointBotConstraints',]#
        os.makedirs(file)#data/SimplePointBot or data/SimplePointBotConstraints
    else:
        raise RuntimeError("Directory %s already exists." % file)#not good code writing
    teacher = teacher(env, noisy=noisy)#SimplePointBotTeacher, or ConstraintTeacher,
    demonstrations = []#an empty list
    for i in range(n):
        traj = teacher.generate_trajectory()#around line 33 in teacher.py
        reward = sum([frame['reward'] for frame in traj])#traj is a list of dictionaries
        #why not directly use rtg[0]?
        print('Trajectory %d, Reward %d' % (i, reward))
        demonstrations.append(traj)#traj is one piece of trajectories
        utils.save_trajectory(traj, file, i)#around line 86 in utils.py#save 1 piece of traj
        if i < 50 and logdir is not None:
            pu.make_movie(traj, os.path.join(logdir, '%s_%d.gif' % (data_dir, i)))
    return demonstrations#list of list of trajectories?

def generate_teacher_demo_datasafety(env, data_dir, teacher, n=100, noisy=False, logdir=None):
    log.info("Generating teacher demo trajectories")
    file = os.path.join('data', data_dir)#create the data folder!
    if not os.path.exists(file):#['SimplePointBot','SimplePointBotConstraints',]#
        os.makedirs(file)#data/SimplePointBot or data/SimplePointBotConstraints
    else:
        raise RuntimeError("Directory %s already exists." % file)#not good code writing
    teacher = teacher(env, noisy=noisy)#SimplePointBotTeacher, or ConstraintTeacher,
    demonstrations = []#an empty list
    for i in range(n):
        traj = teacher.generate_trajectorysafety()#line 33 in teacher.py#100 transitions
        reward = sum([frame['reward'] for frame in traj])#traj is a list of dictionaries
        #why not directly use rtg[0]?
        print('Trajectory %d, Reward %d' % (i, reward))
        demonstrations.append(traj)#traj is one piece of trajectories
        utils.save_trajectory(traj, file, i)#86 in utils.py#save 1 traj having 100 steps
        if i < 50 and logdir is not None:
            pu.make_movie(traj, os.path.join(logdir, '%s_%d.gif' % (data_dir, i)))
    return demonstrations#list of list of trajectories?

def main():
    params = parse_args()

    logdir = utils.get_file_prefix()#around line 46 in utils.py
    os.makedirs(logdir)
    utils.init_logging(logdir)#around line 58 in utils.py

    env = utils.make_env(params)#around line 153#SimplePointBot
    print('horizon',env.horizon)#horizon 100, that is what will be printed!

    teachers = env_teachers[params['env']]#[SimplePointBotTeacher, ConstraintTeacher, StrangeTeacher]
    data_dirs = params['data_dirs']#['SimplePointBot','SimplePointBotConstraints',]
    data_counts = params['data_counts']#[50,50] for spb

    for teacher, data_dir, count in list(zip(teachers, data_dirs, data_counts)):
        #still 2, always take the least number: https://www.programiz.com/python-programming/methods/built-in/zip
        #generate_teacher_demo_data(env, data_dir, teacher, count, True, logdir)#see around 31
        generate_teacher_demo_datasafety(env, data_dir, teacher, count, True, logdir)  # see around 31


if __name__ == '__main__':
    main()
