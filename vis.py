from exp.run_experiment import *
from mplite import TaskManager, Task
import time
import cProfile


def vis():
    # Param def ########################################################################################################
    # params = [np.array([1, wr, 0]) for wr in [0, 2, 4, 6, 8, 10]]  # --------------------------------- Lin, NS, DT-r
    # params = [['diff', 0], ['abs', 2], ['abs', 10], ['diff', 2], ['diff', 10]]  # -------------------- Flickering
    params = [[['allowed', 'center', 0], ['allowed', 'center', 2], ['allowed', 'center', 10]],
              [['allowed', 'left', 0], ['allowed', 'left', 2], ['allowed', 'left', 10]],
              [['forbidden', 'center', 0], ['forbidden', 'center', 2], ['forbidden', 'center', 10]],
              [['forbidden', 'left', 0], ['forbidden', 'left', 2], ['forbidden', 'left', 10]]]  # ------ DT-free
    # params = [np.array([1, 0, wt]) for wt in [0, 2, 4, 6, 8, 10]]  # --------------------------------- Tolman
    # num_runs = 6  # ---------------------------------------------------------------------------------- Tolman

    path = './data/v3/DT-free/'
    img_path = './img/v3/DT-free/'

    # Video plotting ###################################################################################################
    # label = '0'
    # path = path + 'w_10'  # --------------------------------------------------------------------------- most
    # # path = path + 'abs_w_10' # ---------------------------------------------------------------------- Flickering
    # # path = path + 'walls_forbidden_start_left_w_10' # ----------------------------------------------- DT-free
    # experiment_plotter(path, f'environment_{label}.txt', f'agent_{label}.csv', start=95)

    # Saving the video #################################################################################################
    # batches = []
    # label = 0
    # for p in params:
    #     w = p
    #     batches.append(f'w_{w[2]:02d}')
    #     if w[-1] == 0 or w[-1] == 10:
    #         read_path = path + f'w_{w[2]:02d}'
    #         write_path = img_path + f'w_{w[2]:02d}'
    #         experiment_plotter(read_path, f'environment_{label}.txt', f'agent_{label}.csv', start=0, save_path=write_path)

    # Visualizing the rew rate #########################################################################################
    # batches = []
    # for p in params:
        # w = p
        # batches.append(f'w_{w[1]:02d}')
        # batches.append(f'{p[0]}_w_{p[1]:02d}')
        # batches.append(f'w_{w[2]:02d}')
    # rr_plotter(path, batches, img_path=img_path, label='')
    # rr_plotter(path, batches, img_path=img_path, win_begin=95, win_end=105, label='zoomed', log_U=False)

    # Box plotting
    # for setup in params:
    #     batches = [f'walls_{s[0]}_start_{s[1]}_w_{s[2]:02d}' for s in setup]
    #     replay_plotter(path, batches,
    #                    f'walls_{setup[0][0]}_start_{setup[0][1]}_box_norm_begin', [1, 3], 'total', box=True,
    #                    win_end=100, img_path=img_path)
    #     replay_plotter(path, batches,
    #                    f'walls_{setup[0][0]}_start_{setup[0][1]}_box_norm_end', [1, 3], 'total', box=True,
    #                    win_begin=100, img_path=img_path)
    # replay_plotter(path, batches,
    #                'box_norm', [3, 2], 'total', box=True, img_path=img_path)
    #
    # replay_plotter(path, batches,
    #                'box', [3, 2], None, box=True, img_path=img_path)
    #

    #
    # replay_plotter(path, batches,
    #                'box_norm_begin', [3, 2], 'total', box=True, win_end=100, img_path=img_path)
    #

    #
    # replay_plotter(path, batches,
    #                'box_norm_end', [3, 2], 'total', box=True, win_begin=100, img_path=img_path)

    # Maze plotting
    # replay_plotter(path, batches,
    #                'norm', [3, 2], 'total', img_path=img_path)
    #
    # replay_plotter(path, batches,
    #                '', [3, 2], None, img_path=img_path)
    #
    # replay_plotter(path, batches, f'norm_vis_begin', [3, 2], 'total', win_end=100, img_path=img_path, lims=[0, 0.15])
    #
    # replay_plotter(path, batches, f'begin', [3, 2], None, win_end=100, img_path=img_path)
    #
    # replay_plotter(path, batches, f'norm_vis_end', [3, 2], 'total', win_begin=100, img_path=img_path, lims=[0, 0.15])
    #
    # replay_plotter(path, batches, f'end', [3, 2], None, win_begin=100, img_path=img_path)

    # For the Tolman maze
    # for idx_run in range(1, num_runs):
        # replay_plotter(path, batches, f'norm_vis_{idx_run}', [3, 2], 'total', win_begin=idx_run, win_end=idx_run, img_path=img_path)# , lims=np.array([0, 160]))

        # replay_plotter(path, batches, f'vis_{idx_run}_max', [3, 2], None, win_begin=idx_run, win_end=idx_run, img_path=img_path, lims=np.array([0, 100]))


    # For data where batches are 3-dimensional
    for setup in params:
        batches = [f'walls_{s[0]}_start_{s[1]}_w_{s[2]:02d}' for s in setup]
        # rr_plotter(path, batches, img_path=img_path, label=f'walls_{setup[0][0]}_start_{setup[0][1]}')
        replay_plotter(path, batches, f'walls_{setup[0][0]}_start_{setup[0][1]}_norm_vis_begin', [3, 1], 'total', win_end=100, img_path=img_path)#, lims=[0, 0.06])
        replay_plotter(path, batches, f'walls_{setup[0][0]}_start_{setup[0][1]}_begin', [3, 1], None, win_end=100, img_path=img_path)#, lims=[0, 15])
        replay_plotter(path, batches, f'walls_{setup[0][0]}_start_{setup[0][1]}_norm_vis_end', [3, 1], 'total', win_begin=100, img_path=img_path)#, lims=[0, 0.06])
        replay_plotter(path, batches, f'walls_{setup[0][0]}_start_{setup[0][1]}_end', [3, 1], None, win_begin=100, img_path=img_path)#, lims=[0, 15])


    # Comparative plotting
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
    # params = [#['walls_allowed_start_center_w_10', 'walls_allowed_start_left_w_10', 'walls_allowed_w_10'],
    #            # ['walls_allowed_start_center_w_02', 'walls_allowed_start_left_w_02', 'walls_allowed_w_02'],
    #            # ['walls_allowed_start_center_w_00', 'walls_allowed_start_left_w_00', 'walls_allowed_w_00'],
    #            # ['walls_forbidden_start_center_w_10', 'walls_forbidden_start_left_w_10', 'walls_forbidden_w_10'],
    #            # ['walls_forbidden_start_center_w_02', 'walls_forbidden_start_left_w_02', 'walls_forbidden_w_02'],
    #            # ['walls_forbidden_start_center_w_00', 'walls_forbidden_start_left_w_00', 'walls_forbidden_w_00'],
    #            ['walls_allowed_start_center_w_10', 'walls_forbidden_start_center_w_10', 'start_center_w_10'],
    #            ['walls_allowed_start_center_w_02', 'walls_forbidden_start_center_w_02', 'start_center_w_02'],
    #            ['walls_allowed_start_center_w_00', 'walls_forbidden_start_center_w_00', 'start_center_w_00'],
    #            ['walls_forbidden_start_center_w_00', 'walls_forbidden_start_center_w_10', 'walls_forbidden_start_center']] #,
    #            # ['walls_allowed_start_left_w_10', 'walls_forbidden_start_left_w_10', 'start_left_w_10'],
    #            # ['walls_allowed_start_left_w_00', 'walls_forbidden_start_left_w_00', 'start_left_w_00'],
    #            # ['walls_allowed_start_left_w_02', 'walls_forbidden_start_left_w_02', 'start_left_w_02']]
    # for setup in params:
    #     batches = setup[0:2]
    #     legend = setup[2]
    #     replay_plotter(path, batches, f'{legend}_norm_begin', [1, 1], 'total', win_end=100, img_path=img_path, comparative=True) #, lims=[-0.01, 0.01])
    #     # replay_plotter(path, batches, f'{legend}_begin', [1, 1], None, win_end=100, img_path=img_path, comparative=True)
    #     replay_plotter(path, batches, f'{legend}_norm_end', [1, 1], 'total', win_begin=100, img_path=img_path, comparative=True) #, lims=[-0.01, 0.01])
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
