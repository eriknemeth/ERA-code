import numpy as np

from exp.run_experiment import *
from mplite import TaskManager, Task
import time
import cProfile


def main():
    # Param def ########################################################################################################
    weights = [np.array([1, wr, 0]) for wr in [0, 2, 4, 6, 8, 10]]
    er_type = 'diff'
    fw = True
    epsilon = 0.05
    # params = [['allowed', 'center', 0], ['allowed', 'center', 2], ['allowed', 'center', 10],
    #           ['allowed', 'left', 0], ['allowed', 'left', 2], ['allowed', 'left', 10],
    #           ['forbidden', 'center', 0], ['forbidden', 'center', 2], ['forbidden', 'center', 10],
    #           ['forbidden', 'left', 0], ['forbidden', 'left', 2], ['forbidden', 'left', 10]]
    # params = [[0, 0], [0.1, 0], [0.2, 0], [0.4, 0], [0.8, 0], [1, 0]]
    # params = [['diff', 0], ['abs', 2], ['abs', 10], ['diff', 2], ['diff', 10]]
    # params = [['abs', 8], ['diff', 8], ['diff', 0]]
    t = 0.01
    path = './data_v2/DT-restr-v2/'
    max_replay = None
    g = 0.9
    # Parallel execution ###############################################################################################
    tasks = []
    run = os.environ['SLURM_ARRAY_TASK_ID']
    # run = 0
    for w in weights:
    # for p in params:
    #     w = np.array([1, p[2], 0])
    #     batch_name = f'walls_{p[0]}_start_{p[1]}_w_{p[2]:02d}'
    #     batch_name = f'{p[0]}_w_{p[1]:02d}'
        batch_name = f'w_{w[1]:02d}'
        # batch_name = f'eps_{p[0]:.02f}_w_{p[1]:02d}'
        print(batch_name)
        th = t * (w[1] + w[0])
        # er_type = p[0]
        # epsilon = p[0]
        # fw = p[0] == 'forbidden'
        if not os.path.isfile(f'{path}{batch_name}/agent_{run}.csv'):
            ta = Task(free_exploration,
                      *(f'{path}{batch_name}',
                        f'{run}',), **{'dec_weights': w, 'replay_threshold': th,
                                       'prog_bar': False, 'rew_change': 0.5,  # 'beta': b,
                                       'max_replay': max_replay, 'gamma': g,
                                       'maze_type': 'DT', 'forbidden_walls': fw,
                                       'decision_rule': 'epsilon', 'epsilon': epsilon, 'nV': 10, 'restricted': True,
                                       'handle': 'sa', 'known_env': True, 'epist_rew_type': er_type, 'format': 'small',
                                       'flickering': False}) #, 'free_DT_start_from': p[1]})
            tasks.append(ta)
    with TaskManager() as tm:
        tm.execute(tasks)

    # Non-parallel execution ###########################################################################################
    # # run = os.environ['SLURM_ARRAY_TASK_ID']
    # run = 13
    # for p in params:
    #     w = np.array([1, p[2], 0])
    #     batch_name = f'walls_{p[0]}_start_{p[1]}_w_{p[2]:02d}'
    # #     batch_name = f'{p[0]}_w_{p[1]:02d}'
    # #     batch_name = f'w_{w[1]:02d}'
    #     # batch_name = f'eps_{p[0]:.02f}_w_{p[1]:02d}'
    #     print(batch_name)
    #     th = t * (w[1] + w[0])
    #     # er_type = p[0]
    #     # epsilon = p[0]
    #     fw = p[0] == 'forbidden'
    #     if not os.path.isfile(f'{path}{batch_name}/agent_{run}.csv'):
    #         # if not os.path.isfile(f'{path}w_{w[1]:02d}/agent_{run}.csv'):
    #         free_exploration(f'{path}{batch_name}', f'{run}',
    #                          dec_weights=w, replay_threshold=th, rew_change=0.5, max_replay=max_replay,
    #                          prog_bar=True, known_env=True, maze_type='DT', forbidden_walls=fw,
    #                          gamma=g, decision_rule='epsilon', nV=10, restricted=False, handle='sa',
    #                          epist_rew_type=er_type, format='full', flickering=False, free_DT_start_from=p[1])


if __name__ == '__main__':
    # cProfile.run('main()')
    main()
