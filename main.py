import numpy as np

from exp.run_experiment import *
from mplite import TaskManager, Task
import time
import cProfile


def main():
    # Param def ########################################################################################################
    params = [np.array([1, wr, 0]) for wr in [0, 2, 4, 6, 8, 10]]  # ----------------------------------- Lin, NS, DT-r
    # params = [['diff', 0], ['abs', 2], ['abs', 10], ['diff', 2], ['diff', 10]]  # -------------------- Flickering
    # params = [['allowed', 'center', 0], ['allowed', 'center', 2], ['allowed', 'center', 10],
    #           ['allowed', 'left', 0], ['allowed', 'left', 2], ['allowed', 'left', 10],
    #           ['forbidden', 'center', 0], ['forbidden', 'center', 2], ['forbidden', 'center', 10],
    #           ['forbidden', 'left', 0], ['forbidden', 'left', 2], ['forbidden', 'left', 10]]  # ------ DT-free
    # params = [np.array([1, 0, wt]) for wt in [0, 2, 4, 6, 8, 10]]  # --------------------------------- Tolman
    # ------------------------------------------------------------------------------------------------------------------
    model_type = 'MB'
    replay_type = 'priority'
    epsilon = 0.05
    t = 0.01
    max_replay = None
    g = 0.9

    maze_type = 'DT'  # -------------------------------------------------------------------------------- env
    nV = 2  # ------------------------------------------------------------------------------------------ model window
    runs = 200  # -------------------------------------------------------------------------------------- duration
    rew_change = 0.5 # --------------------------------------------------------------------------------- timing

    er_type = 'diff'  # -------------------------------------------------------------------------------- Flickering
    flick = False  # ----------------------------------------------------------------------------------- Flickering
    restr = True  # ------------------------------------------------------------------------------------ DT-restr
    fw = True  # --------------------------------------------------------------------------------------- DT-free, Tolman
    # wall_loc = np.array([[9, 12]]) # ----------------------------------------------------------------- Tolman
    # wall_change = 1/2  # ----------------------------------------------------------------------------- Tolman
    # new_wall_loc = np.array([[1, 5], [9, 12]])  # ---------------------------------------------------- Tolman

    path = './data/v3/DT-restr/'
    ####################################################################################################################
    try:
        # Parallel execution ###########################################################################################
        run = os.environ['SLURM_ARRAY_TASK_ID']
        tasks = []
        for p in params:
            w = p
            # w = np.array([1, p[1], 0])  # ----------------------------------------------------------------- Flickering
            # w = np.array([1, p[2], 0])  # ----------------------------------------------------------------- DT-free

            th = t * (w[1] + w[0])  # ----------------------------------------------------------------------- most exp
            # th = t * (w[2] + w[0])  # --------------------------------------------------------------------- Tolman

            batch_name = f'w_{w[1]:02d}'
            # batch_name = f'{p[0]}_w_{p[1]:02d}' # --------------------------------------------------------- Flickering
            # batch_name = f'walls_{p[0]}_start_{p[1]}_w_{p[2]:02d}' # -------------------------------------- DT-free
            # batch_name = f'w_{w[2]:02d}'  # --------------------------------------------------------------- Tolman
            print(batch_name)

            # er_type = p[0]  # ----------------------------------------------------------------------------- Flickering
            # fw = p[0] == 'forbidden'  # ------------------------------------------------------------------- DT-free

            if not os.path.isfile(f'{path}{batch_name}/agent_{run}.csv'):
                ta = Task(free_exploration,
                          *(f'{path}{batch_name}',
                            f'{run}',), **{'known_env': True, 'maze_type': maze_type, 'num_runs': runs,
                                           'forbidden_walls': fw, 'flickering': flick, 'restricted': restr,
                                           'rew_change': rew_change,
                                           'model_type': model_type, 'gamma': g, 'nV': nV, 'epist_rew_type': er_type,
                                           'dec_weights': w,
                                           'decision_rule': 'epsilon', 'epsilon': epsilon,
                                           'max_replay': max_replay, 'replay_type': replay_type,
                                           'replay_threshold': th, 'handle': 'sa',
                                           'prog_bar': False, 'format': 'small'})  #,
                                           # 'free_DT_start_from': p[1]})
                                           # 'wall_loc': wall_loc, 'wall_change': wall_change, 'new_wall_loc': new_wall_loc,
                                           # 'num_visits_training': 200, 'state_of_interest': 5, 'action_of_interest': 1})
                tasks.append(ta)
        with TaskManager() as tm:
            tm.execute(tasks)
    except KeyError:
        # Non-parallel execution #######################################################################################
        run = 0
        for p in params:
            w = p
            # w = np.array([1, p[1], 0])  # ---------------------------------------------------------------- Flickering
            # w = np.array([1, p[2], 0])  # ---------------------------------------------------------------- DT-free

            th = t * (w[1] + w[0])  # ---------------------------------------------------------------------- most exp
            # th = t * (w[2] + w[0])  # -------------------------------------------------------------------- Tolman

            batch_name = f'w_{w[1]:02d}'
            # batch_name = f'{p[0]}_w_{p[1]:02d}' # -------------------------------------------------------- Flickering
            # batch_name = f'walls_{p[0]}_start_{p[1]}_w_{p[2]:02d}' # ------------------------------------- DT-free
            # batch_name = f'w_{w[2]:02d}'  # -------------------------------------------------------------- Tolman
            print(batch_name)

            # er_type = p[0]  # ---------------------------------------------------------------------------- Flickering
            # fw = p[0] == 'forbidden'  # ------------------------------------------------------------------ DT-free

            if not os.path.isfile(f'{path}{batch_name}/agent_{run}.csv'):
                free_exploration(f'{path}{batch_name}', f'{run}',
                                 known_env=True, maze_type=maze_type, num_runs=runs,
                                 forbidden_walls=fw, flickering=flick, restricted=restr,
                                 rew_change=rew_change,
                                 model_type=model_type, gamma=g, nV=nV, epist_rew_type=er_type,
                                 dec_weights=w,
                                 decision_rule='epsilon', epsilon=epsilon,
                                 max_replay=max_replay, replay_type=replay_type,
                                 replay_threshold=th, handle='sa',
                                 prog_bar=True, format='full')  #,
                                 # free_DT_start_from = p[1])
                                 # wall_loc=wall_loc, wall_change=wall_change, new_wall_loc=new_wall_loc,
                                 # num_visits_training=200, state_of_interest=5, action_of_interest=1)


if __name__ == '__main__':
    # cProfile.run('main()')
    main()
