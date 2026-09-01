from exp.run_experiment import *
from mplite import TaskManager, Task
import time
import cProfile


def compute_stats():
    # Param def ########################################################################################################
    params = [np.array([1, wr, 0]) for wr in [0, 2, 4, 6, 8, 10]]
    path = './data/v3/DT-restr/'
    stat_path = './stat/v3/DT-restr/'

    batches = [f'w_{p[1]:02d}' for p in params]
    replay_stats(path, batches, 'total', restricted=True, to_plot='content', stat_path=stat_path, win_end=100, label=f'CONTENT_begin_zoomed', all_rew_locs=False, lin_fit_window=[1, 7])
    replay_stats(path, batches, 'total', restricted=True, to_plot='content', stat_path=stat_path, win_begin=100, label=f'CONTENT_end_both_rew_zoomed', all_rew_locs=True, lin_fit_window=[1,7])

    # ==================================================================================================================
    # params = [[['allowed', 'center', 0], ['allowed', 'center', 2], ['allowed', 'center', 10]],
    #           [['allowed', 'left', 0], ['allowed', 'left', 2], ['allowed', 'left', 10]],
    #           [['forbidden', 'center', 0], ['forbidden', 'center', 2], ['forbidden', 'center', 10]],
    #           [['forbidden', 'left', 0], ['forbidden', 'left', 2], ['forbidden', 'left', 10]]]
    # path = './data/v3/DT-free/'
    # stat_path = './stat/v3/DT-free/'
    #
    # for setup in params:
    #     batches = [f'walls_{s[0]}_start_{s[1]}_w_{s[2]:02d}' for s in setup]
    #     replay_stats(path, batches, None, restricted=False, to_plot='loc', stat_path=stat_path, win_end=100,
    #                  label=f'LOC_walls_{setup[0][0]}_start_{setup[0][1]}_begin_zoomed', all_rew_locs=False, lin_fit_window=[1, 7])
    #     replay_stats(path, batches, None, restricted=False, to_plot='loc', stat_path=stat_path, win_begin=100,
    #                  label=f'LOC_walls_{setup[0][0]}_start_{setup[0][1]}_end_both_rew_locs_zoomed', all_rew_locs=True, lin_fit_window=[1, 7])


if __name__ == '__main__':
    compute_stats()
