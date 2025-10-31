from exp.run_experiment import *
from mplite import TaskManager, Task
import time
import cProfile


def compute_stats():
    # Param def ########################################################################################################
    # weights = [np.array([1, wr, 0]) for wr in [0, 2, 4, 6, 8, 10]]
    # weights = [np.array([1, wr, 0]) for wr in [8, 9, 10, 11, 12]]
    # params = [['diff', 0], ['abs', 2], ['abs', 10], ['diff', 2], ['diff', 10]]
    # params = [[0, 0], [0.05, 0], [0.1, 0], [0.2, 0], [0.4, 0], [0.8, 0], [1, 0]]
    params = [[['allowed', 'center', 0], ['allowed', 'center', 2], ['allowed', 'center', 10]],
              [['allowed', 'left', 0], ['allowed', 'left', 2], ['allowed', 'left', 10]],
              [['forbidden', 'center', 0], ['forbidden', 'center', 2], ['forbidden', 'center', 10]],
              [['forbidden', 'left', 0], ['forbidden', 'left', 2], ['forbidden', 'left', 10]]]
    path = './data/DT-free-v2/'
    # path = './data_v2/DT-free-v2/'
    stat_path = './stat/'

    batches = []
    # for w in weights:
    #     batches.append(f'w_{w[1]:02d}')
    # for p in params:
    #     batches.append(f'{p[0]}_w_{p[1]:02d}')
    # for p in params:
    #     batches.append(f'eps_{p[0]:.02f}_w_{p[1]:02d}')

    # replay_stats(path, batches, 'total', stat_path=stat_path, win_end=100, label='begin')
    # replay_stats(path, batches, 'total', stat_path=stat_path, win_begin=100, label='end')

    for setup in params:
        batches = [f'walls_{s[0]}_start_{s[1]}_w_{s[2]:02d}' for s in setup]
        replay_stats(path, batches, 'total', stat_path=stat_path, label=f'walls_{setup[0][0]}_start_{setup[0][1]}')
        replay_stats(path, batches, 'total', stat_path=stat_path, win_end=100, label=f'walls_{setup[0][0]}_start_{setup[0][1]}_begin')
        replay_stats(path, batches, 'total', stat_path=stat_path, win_begin=100, label=f'walls_{setup[0][0]}_start_{setup[0][1]}_end')


if __name__ == '__main__':
    compute_stats()
