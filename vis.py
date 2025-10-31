from exp.run_experiment import *
from mplite import TaskManager, Task
import time
import cProfile


def vis():
    # Param def ########################################################################################################
    weights = [np.array([1, wr, 0]) for wr in [0, 2, 4, 6, 8, 10]]
    # weights = [np.array([1, wr, 0]) for wr in [8, 9, 10, 11, 12]]
    # params = [['diff', 0], ['abs', 2], ['abs', 10], ['diff', 2], ['diff', 10]]
    # params = [[0, 0], [0.05, 0], [0.1, 0], [0.2, 0], [0.4, 0], [0.8, 0], [1, 0]]
    # params = [[['allowed', 'center', 0], ['allowed', 'center', 2], ['allowed', 'center', 10]],
    #           [['allowed', 'left', 0], ['allowed', 'left', 2], ['allowed', 'left', 10]],
    #           [['forbidden', 'center', 0], ['forbidden', 'center', 2], ['forbidden', 'center', 10]],
    #           [['forbidden', 'left', 0], ['forbidden', 'left', 2], ['forbidden', 'left', 10]]]
    path = './data_v2/DT-restr-v2/'
    img_path = './img_v2/DT-restr-v2/'
    # Video plotting ###################################################################################################
    # label = '0'
    # # path = path + 'walls_forbidden_start_center_w_02'
    # # path = path + 'eps_1.00_w_00'
    # path = path + 'w_00'
    # # path = path + 'abs_w_10'
    # experiment_plotter(path, f'environment_{label}.txt', f'agent_{label}.csv', start=98)
    # # Matrix visualization ###########################################################################################
    # print('Plotting...')
    # batches = []
    # for w in weights:
    #     for t in thresh:
    #         batches.append(f'w_{w[1]:.1f}_t_{t:.3f}')
    # matrix_plotter(path, ['w', 't'],
    #                [['w', 't']], batches, methods=['max', 'mean'])
    #
    # matrix_plotter(path, ['w', 't'],
    #                [['w', 't']], batches, methods=['max', 'mean'],
    #                win_begin=0, win_end=ceil(200 * 1/3), label='before')
    #
    # matrix_plotter(path, ['w', 't'],
    #                [['w', 't']], batches, methods=['max', 'mean'],
    #                win_begin=50, win_end=None, label='end')

    # Visualizing the rew rate #########################################################################################
    # for w in weights:
    #     batches = []
    #     for t in thresh:
    #         batches.append(f'w_{w[1]:.1f}_t_{t:.3f}')
    #     cumulative_plotter(path, batches,
    #                        f'w_{w[1]:.1f}', [3, 2], True, img_path=img_path)

    # for t in thresh:
    # weights = [np.array([1, wr, 0]) for wr in [6, 8, 10]]
    batches = []
    # for p in params:
    #     batches.append(f'{p[0]}_w_{p[1]:02d}')
    # for p in params:
    #     batches.append(f'eps_{p[0]:.02f}_w_{p[1]:02d}')
    for w in weights:
        batches.append(f'w_{w[1]:02d}')
    rr_plotter(path, batches, img_path=img_path, label='')
    # rr_plotter(path, batches, img_path=img_path, win_begin=95, win_end=105, label='zoomed', log_U=False)
    # weights = [np.array([1, wr, 0]) for wr in [0, 2, 4, 6, 8, 10]]

    # Box plotting
    # replay_plotter(path, batches,
    #                'box_norm', [3, 2], 'total', box=True, img_path=img_path)
    #
    # replay_plotter(path, batches,
    #                'box', [3, 2], None, box=True, img_path=img_path)
    #
    # replay_plotter(path, batches,
    #                'box_norm_begin', [3, 2], 'total', box=True, win_end=100, img_path=img_path)
    #
    # replay_plotter(path, batches,
    #                'box_begin', [3, 2], None, box=True, win_end=100, img_path=img_path)
    #
    # replay_plotter(path, batches,
    #                'box_norm_end', [3, 2], 'total', box=True, win_begin=100, img_path=img_path)
    #
    # replay_plotter(path, batches,
    #                'box_end', [3, 2], None, box=True, win_begin=100, img_path=img_path)

    # Maze plotting
    # replay_plotter(path, batches,
    #                'norm', [3, 2], 'total', img_path=img_path)
    #
    # replay_plotter(path, batches,
    #                '', [3, 2], None, img_path=img_path)
    #
    replay_plotter(path, batches, f'norm_vis_begin', [3, 2], 'total', win_end=100, img_path=img_path, lims=[0, 0.15])
    #
    # replay_plotter(path, batches, f'begin', [3, 2], None, win_end=100, img_path=img_path)
    #
    replay_plotter(path, batches, f'norm_vis_end', [3, 2], 'total', win_begin=100, img_path=img_path, lims=[0, 0.15])
    #
    # replay_plotter(path, batches, f'end', [3, 2], None, win_begin=100, img_path=img_path)

    # For data where batches are 3-dimensional
    # for setup in params:
    #     batches = [f'walls_{s[0]}_start_{s[1]}_w_{s[2]:02d}' for s in setup]
    #     # rr_plotter(path, batches, img_path=img_path, label=f'walls_{setup[0][0]}_start_{setup[0][1]}')
    #     replay_plotter(path, batches, f'walls_{setup[0][0]}_start_{setup[0][1]}_norm_vis_begin', [3, 1], 'total', win_end=100, img_path=img_path)
    #     # replay_plotter(path, batches, f'walls_{setup[0][0]}_start_{setup[0][1]}_begin', [3, 1], None, win_end=100, img_path=img_path)
    #     replay_plotter(path, batches, f'walls_{setup[0][0]}_start_{setup[0][1]}_norm_vis_end', [3, 1], 'total', win_begin=100, img_path=img_path)
    #     # replay_plotter(path, batches, f'walls_{setup[0][0]}_start_{setup[0][1]}_end', [3, 1], None, win_begin=100, img_path=img_path)


    # Comparative plotting
    # batches = [f'w_{weights[0][1]:02d}', f'w_{weights[-1][1]:02d}']
    #
    # Maze-like
    # replay_plotter(path, batches,
    #                'norm', [1, 1], 'total', img_path=img_path, comparative=True)
    #
    # replay_plotter(path, batches,
    #                '', [1, 1], None, img_path=img_path, comparative=True)
    #
    # replay_plotter(path, batches, f'norm_begin', [1, 1], 'total', win_end=100, img_path=img_path, comparative=True)
    #
    # replay_plotter(path, batches, f'begin', [1, 1], None, win_end=100, img_path=img_path, comparative=True)
    #
    # replay_plotter(path, batches, f'norm_end', [1, 1], 'total', win_begin=100, img_path=img_path, comparative=True)
    #
    # replay_plotter(path, batches, f'end', [1, 1], None, win_begin=100, img_path=img_path, comparative=True)

    # For data where batches are 3-dimensional
    # params = [['walls_allowed_start_center_w_10', 'walls_allowed_start_left_w_10', 'walls_allowed_w_10'],
    #            ['walls_allowed_start_center_w_02', 'walls_allowed_start_left_w_02', 'walls_allowed_w_02'],
    #            ['walls_allowed_start_center_w_00', 'walls_allowed_start_left_w_00', 'walls_allowed_w_00'],
    #            ['walls_forbidden_start_center_w_10', 'walls_forbidden_start_left_w_10', 'walls_forbidden_w_10'],
    #            ['walls_forbidden_start_center_w_02', 'walls_forbidden_start_left_w_02', 'walls_forbidden_w_02'],
    #            ['walls_forbidden_start_center_w_00', 'walls_forbidden_start_left_w_00', 'walls_forbidden_w_00'],
    #            ['walls_allowed_start_center_w_10', 'walls_forbidden_start_center_w_10', 'start_center_w_10'],
    #            ['walls_allowed_start_center_w_02', 'walls_forbidden_start_center_w_02', 'start_center_w_02'],
    #            ['walls_allowed_start_center_w_00', 'walls_forbidden_start_center_w_00', 'start_center_w_00']] #,
    #            # ['walls_allowed_start_left_w_10', 'walls_forbidden_start_left_w_10', 'start_left_w_10'],
    #            # ['walls_allowed_start_left_w_00', 'walls_forbidden_start_left_w_00', 'start_left_w_00']]
    # for setup in params:
    #     batches = setup[0:2]
    #     legend = setup[2]
    #     replay_plotter(path, batches, f'{legend}_norm_begin', [1, 1], 'total', win_end=100, img_path=img_path, comparative=True, lims=[-0.35, 0.35])
    #     # replay_plotter(path, batches, f'{legend}_begin', [1, 1], None, win_end=100, img_path=img_path, comparative=True)
    #     replay_plotter(path, batches, f'{legend}_norm_end', [1, 1], 'total', win_begin=100, img_path=img_path, comparative=True, lims=[-0.35, 0.35])
    #     # replay_plotter(path, batches, f'{legend}_end', [1, 1], None, win_begin=100, img_path=img_path, comparative=True)

    # Bar-like
    # replay_plotter(path, batches,
    #                'norm', [1, 1], 'total', img_path=img_path, comparative=True, bar=True)
    #
    # replay_plotter(path, batches,
    #                '', [1, 1], None, img_path=img_path, comparative=True, bar=True)
    #
    # replay_plotter(path, batches, f'norm_begin', [1, 1], 'total', win_end=100, img_path=img_path, comparative=True, bar=True)
    #
    # replay_plotter(path, batches, f'begin', [1, 1], None, win_end=100, img_path=img_path, comparative=True, bar=True)
    #
    # replay_plotter(path, batches, f'norm_end', [1, 1], 'total', win_begin=100, img_path=img_path, comparative=True, bar=True)
    #
    # replay_plotter(path, batches, f'end', [1, 1], None, win_begin=100, img_path=img_path, comparative=True, bar=True)

    # Visualizing replayed chunks ######################################################################################
    # chunks = [0, 30, 60, 90, 120, 150, 180]
    # for w in weights:
    #     batch = f'w_{w[1]:02d}'
    #     cumulative_plotter_chunks(path, batch, batch, [2, 3], chunks, img_path=img_path)


if __name__ == '__main__':
    vis()
