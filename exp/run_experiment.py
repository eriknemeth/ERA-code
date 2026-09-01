from classes.RLA import *
from classes.plotter import *
from tqdm import tqdm
import re
from pympler.asizeof import asizeof

def replay_stats(path: str, batches: list, norm_rep, restricted: bool, to_plot: str,
                   **kwargs) -> None:
    win_begin = kwargs.get('win_begin', 0)
    win_end = kwargs.get('win_end', None)
    stat_path = kwargs.get('stat_path', './stat')
    starting = True
    dTm = None
    label = kwargs.get('label', '')
    regex = re.compile('agent.*csv')
    for batch in batches:
        curr_path = f'{path}{batch}/'
        print(f'Batch {batch}...')
        for root, subdirs, files in os.walk(curr_path):
            for file in tqdm(files):
                if regex.match(file):
                    env_file = f'environment{file[5:-4]}.txt'
                    if starting:
                        dTm = PlotterEnv(env_file, use_epochs=False, with_replay=False, win_size=100,
                                         path=f'{path}{batches[0]}/', norm_rep=norm_rep, win_begin=win_begin,
                                         win_end=win_end)
                        starting = False
                    dTm.load_events(file, env_file, batch, path=curr_path)
    print('Computing stats...')
    dTm.DT_compare_left_right(to_plot, batches, stat_path, label=label)
    dTm.DT_plot_biases(to_plot, batches, stat_path, restricted=restricted, label=label,
                       all_rew_locs=kwargs.get('all_rew_locs', False), lin_fit_window=kwargs.get('lin_fit_window', None))



def rr_plotter(path: str, batches: list, **kwargs) -> None:
    label = kwargs.get('label', '')
    img_path = kwargs.get('img_path', './img')
    win_begin = kwargs.get('win_begin', 0)
    win_end = kwargs.get('win_end', None)
    log_U = kwargs.get('log_U', True)
    regex = re.compile('agent.*csv')
    starting = True
    dTm = None
    # dTm_size = 0
    for batch in batches:
        curr_path = f'{path}{batch}/'
        print(f'Batch {batch}...')
        for root, subdirs, files in os.walk(curr_path):
            for file in tqdm(files):
                if regex.match(file):
                    env_file = f'environment{file[5:-4]}.txt'
                    if starting:
                        dTm = PlotterEnv(env_file, use_epochs=False, with_replay=False, win_size=100,
                                         path=f'{path}{batches[0]}/', win_begin=win_begin, win_end=win_end)
                        starting = False
                        # dTm_size = asizeof(dTm)
                    dTm.load_events(file, env_file, batch, path=curr_path)
                    # print(f'Size of environment:\n\t- before: {dTm_size} bytes\n\t- after: {asizeof(dTm)} bytes\n\t\t- data: {asizeof(dTm)-dTm_size}')
                    # dTm_size = asizeof(dTm)
    print('Plotting...')
    dTm.plot_reward_rates(batches, save_img=True, path=img_path, label=label, U_rates=True)
    plt.close()
    dTm.plot_U_dynamics(batches, save_img=True, path=img_path, label=label, U_rates=True, log_U=log_U)
    # plt.pause(5)
    plt.close()
    # dTm.plot_rep_vs_visits(batches, save_img=True, path=img_path, label=label)
    # plt.close()


def replay_plotter(path: str, batches: list, label: str, fig_shape: list, norm_rep,
                   **kwargs) -> None:
    win_begin = kwargs.get('win_begin', 0)
    win_end = kwargs.get('win_end', None)
    img_path = kwargs.get('img_path', './img')
    comp = kwargs.get('comparative', False)
    if comp and len(batches) != 2:
        raise ValueError("Only two batches can be compared")
    starting = True
    dTm = None
    regex = re.compile('agent.*csv')
    for batch in batches:
        curr_path = f'{path}{batch}/'
        print(f'Batch {batch}...')
        for root, subdirs, files in os.walk(curr_path):
            for file in tqdm(files):
                if regex.match(file):
                    env_file = f'environment{file[5:-4]}.txt'
                    if starting:
                        dTm = PlotterEnv(env_file, use_epochs=False, with_replay=False, win_size=100,
                                         path=f'{path}{batches[0]}/', norm_rep=norm_rep, win_begin=win_begin,
                                         win_end=win_end)
                        starting = False
                    dTm.load_events(file, env_file, batch, path=curr_path)
    print('Plotting...')
    if not comp:
        # dTm.plot_replay('loc', batches, fig_shape, save_img=True, path=img_path, label=label,
        #                 box_plot=kwargs.get('box', False), lims=kwargs.get('lims', None))
        # plt.close()
        dTm.plot_replay('content', batches, fig_shape, save_img=True, path=img_path, label=label,
                        box_plot=kwargs.get('box', False), lims=kwargs.get('lims', None))
        plt.close()
        # dTm.plot_replay('stationed', batches, fig_shape, save_img=True, path=img_path, label=label,
        #                 box_plot=kwargs.get('box', False), lims=kwargs.get('lims', None))
        # plt.close()
    else:
        # dTm.plot_replay_comp('loc', batches, save_img=True, path=img_path, label=label, bar=kwargs.get('bar', False),
        #                      lims=kwargs.get('lims', None))
        # plt.close()
        dTm.plot_replay_comp('content', batches, save_img=True, path=img_path, label=label,
                             bar=kwargs.get('bar', False), lims=kwargs.get('lims', None))
        plt.close()
        # dTm.plot_replay_comp('stationed', batches, save_img=True, path=img_path, label=label,
        #                      bar=kwargs.get('bar', False), lims=kwargs.get('lims', None))
        # plt.close()


def cumulative_plotter_chunks(path: str, batch: str, label: str, fig_shape: list,
                              replay_chunks: list, **kwargs) -> None:
    img_path = kwargs.get('img_path', './img')
    starting = True
    dTm = None
    regex = re.compile('agent.*csv')
    curr_path = f'{path}{batch}/'
    print(f'Batch {batch}...')
    for root, subdirs, files in os.walk(curr_path):
        for file in tqdm(files):
            if regex.match(file):
                env_file = f'environment{file[5:-4]}.txt'
                if starting:
                    dTm = PlotterEnv(env_file, use_epochs=False, with_replay=False, win_size=100,
                                     path=f'{path}{batch}/')
                    starting = False
                dTm.load_events(file, env_file, batch, path=curr_path, chunked_replay=replay_chunks)
    print('Plotting...')
    dTm.plot_replay('content', replay_chunks[0:-1], fig_shape, save_img=True, path=img_path, label=label)
    plt.close()


def matrix_plotter(path: str, axes: list, axes_to_plot: list, batches: list, **kwargs) -> None:
    win_begin = kwargs.get('win_begin', 0)
    win_end = kwargs.get('win_end', None)
    img_path = kwargs.get('img_path', './img')
    starting = True
    dTm = None
    regex = re.compile('agent.*csv')
    for batch in batches:
        curr_path = f'{path}{batch}/'
        print(f'Batch {batch}...')
        for root, subdirs, files in os.walk(curr_path):
            for file in tqdm(files):
                if regex.match(file):
                    env_file = f'environment{file[5:-4]}.txt'
                    if starting:
                        dTm = PlotterEnv(env_file, use_epochs=False, with_replay=False, win_size=100,
                                         path=f'{path}{batches[0]}/',
                                         params=axes, win_begin=win_begin, win_end=win_end)
                        starting = False
                    dTm.load_events(file, env_file, batch, path=curr_path)
    print('Plotting...')
    label = kwargs.get('label', '')
    methods = kwargs.get('methods', 'max')
    for ax in axes_to_plot:
        for meth in methods:
            dTm.plot_cumul_rew_matrix(ax, save_img=True, path=img_path, label=label, method=meth)


def experiment_plotter(path: str, env_file: str, agent_file: str, **kwargs):
    """
    Plots the data gathered from a specific experiment
    :param path: path to the experimental data
    :param env_file: name of the environment file [.txt]
    :param agent_file: name of the agent events file [.csv]
    :param kwargs:
        weights: weights to combine the Q, Ur and Ut values into C-values [np.ndarray]
        save_path: where to save the plot image by image
    :return:
    """
    dTm = PlotterEnv(env_file, path=path)
    dTm.load_events(agent_file, env_file, 'MB', path=path)
    dTm.plot_events(start=kwargs.get('start', 0), save_path=kwargs.get('save_path', None))
    plt.close()


def free_exploration(save_path: str, tag: str, dec_weights, **kwargs) -> None:
    """
    Based on Massi et al.
    double reward maze, free motion
    :return:
    """

    # The parameters I have tested
    prog_bar = kwargs.get('prog_bar', False)
    rep_weights = kwargs.get('rep_weights', dec_weights)
    replay_threshold = kwargs.get('replay_threshold', 0.02)
    decision_rule = kwargs.get('decision_rule', 'softmax')
    add_predecessors = kwargs.get('add_predecessors', 'both')
    rew_change = kwargs.get('rew_change', 1 / 2)
    forbidden_walls = kwargs.get('forbidden_walls', False)
    model_type = kwargs.get('model_type', 'MB')
    max_replay = kwargs.get('max_replay', 50)
    known_env = kwargs.get('known_env', True)
    replay_type = kwargs.get('replay_type', 'priority')
    beta = kwargs.get('beta', 50)
    g = kwargs.get('gamma', 0.9)
    maze_type = kwargs.get('maze_type', 'DT')
    nV = kwargs.get('nV', 4)
    restricted = kwargs.get('restricted', False)
    handle = kwargs.get('handle', 'sa')
    save_format = kwargs.get('format', 'full')
    epist_rew_type = kwargs.get('epist_rew_type', 'diff')
    flickering = kwargs.get('flickering', False)
    epsilon = kwargs.get('epsilon', 0.05)
    free_DT_start_from = kwargs.get('free_DT_start_from', 'center')
    wall_loc = kwargs.get('wall_loc', None)
    wall_change = kwargs.get('wall_change', None)
    new_wall_loc = kwargs.get('new_wall_loc', None)
    num_visits_training = kwargs.get('num_visits_training', None)
    state_of_interest = kwargs.get('state_of_interest', None)
    action_of_interest = kwargs.get('action_of_interest', None)
    num_runs = kwargs.get('num_runs', 200)



    # Creating a model
    params = dict()

    # About saving
    params['prog_bar'] = prog_bar  # ------------------------ Progress bar while runnig?
    params['save_data'] = True  # --------------------------- Should save the steps taken into a csv?
    if params['save_data']:
        params['save_path'] = save_path  # --------------- Where should I save
        params['save_tag'] = tag  # ------------------------- What tag should I put on saved data
        params['save_format'] = save_format  # ---------------- full or small

    # About the maze
    params['maze_type'] = maze_type  # --------------------------- The maze type (DT = double T, M = mitochondria)
    if params['maze_type'] == 'DT':
        params['teleport'] = True  # ------------------------ The agent teleports after getting the reward [True] or not
        params['start_pos'] = 20  # ----------------------------- What state do we start from
        params['restricted_dT'] = restricted  # ---------------------- Is the movement restricted to unidirectional?
        if not params['restricted_dT']:
            if free_DT_start_from == 'left':
                params['start_pos'] = 0
            params[
                'episode_length'] = 25  # --------------------------- 10 steps should be enough to reach the distal reward
            # Old reward locations:
            # params['rew_loc'] = np.array([19, 10])  # --------------- What is (are) the rewarded state(s)
            # params['rew_val'] = np.array([0.5, 3.5])  # ----------------- What is (are) the value(s) of the reward(s)
            # params['rew_prob'] = np.array(
            #     [1, 1])  # ---------------- What is (area) the probability/ies of the reward(s)
            # params['new_rew_loc'] = np.array([8])  # ------------------ What is (are) the rewarded state(s)
            # params['new_rew_val'] = np.array([3])  # ------------------ What is (are) the value(s) of the reward(s)
            # params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
            # New reward locations:
            params['rew_loc'] = np.array([19])  # --------------- What is (are) the rewarded state(s)
            params['rew_val'] = np.array([1])  # ----------------- What is (are) the value(s) of the reward(s)
            params['rew_prob'] = np.array([1])  # ---------------- What is (area) the probability/ies of the reward(s)
            params['new_rew_loc'] = np.array([21])  # ------------------ What is (are) the rewarded state(s)
            params['new_rew_val'] = np.array([1])  # ------------------ What is (are) the value(s) of the reward(s)
            params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
        else:
            params[
                'episode_length'] = None  # --------------------------- 10 steps should be enough to reach the distal reward
            # params['rew_loc'] = np.array([19])  # --------------- What is (are) the rewarded state(s)
            params['rew_loc'] = np.array([22])  # --------------- What is (are) the rewarded state(s)
            params['rew_val'] = np.array([1])  # ----------------- What is (are) the value(s) of the reward(s)
            params['rew_prob'] = np.array([1])  # ---------------- What is (area) the probability/ies of the reward(s)
            # params['new_rew_loc'] = np.array([21])  # ------------------ What is (are) the rewarded state(s)
            params['new_rew_loc'] = np.array([29])  # ------------------ What is (are) the rewarded state(s)
            params['new_rew_val'] = np.array([1])  # ------------------ What is (are) the value(s) of the reward(s)
            params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
    elif params['maze_type'] == 'M':
        params['start_pos'] = 57  # ----------------------------- What state do we start from
        params['rew_loc'] = np.array([101, 28])  # --------------- What is (are) the rewarded state(s)
        params['rew_val'] = np.array([1, 20])  # ----------------- What is (are) the value(s) of the reward(s)
        params['rew_prob'] = np.array([1, 1])  # ---------------- What is (area) the probability/ies of the reward(s)
        params['new_rew_loc'] = np.array([38])  # ------------------ What is (are) the rewarded state(s)
        params['new_rew_val'] = np.array([20])  # ------------------ What is (are) the value(s) of the reward(s)
        params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
        params[
            'episode_length'] = 48  # ---------------------------- 32 steps should be enough to reach the distal reward
    elif params['maze_type'] == 'Open':
        params['x_dim'] = 38
        params['y_dim'] = 50
        params['start_pos'] = 1612  # ----------------------------- What state do we start from
        params['rew_loc'] = np.array([1405, 1644])  # --------------- What is (are) the rewarded state(s)
        params['rew_val'] = np.array([1, 20])  # ----------------- What is (are) the value(s) of the reward(s)
        params['rew_prob'] = np.array([1, 1])  # ---------------- What is (area) the probability/ies of the reward(s)
        params['new_rew_loc'] = np.array([278])  # ------------------ What is (are) the rewarded state(s)
        params['new_rew_val'] = np.array([20])  # ------------------ What is (are) the value(s) of the reward(s)
        params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
        params[
            'episode_length'] = 65  # ---------------------------- 43 steps should be enough to reach the distal reward
    elif params['maze_type'] == 'ED':
        params['start_pos'] = 102  # ----------------------------- What state do we start from
        params['rew_loc'] = np.array([87, 94])  # --------------- What is (are) the rewarded state(s)
        params['rew_val'] = np.array([0.375, 6.25])  # ----------------- What is (are) the value(s) of the reward(s)
        params['rew_prob'] = np.array([1, 1])  # ---------------- What is (area) the probability/ies of the reward(s)
        params['new_rew_loc'] = np.array([6])  # ------------------ What is (are) the rewarded state(s)
        params['new_rew_val'] = np.array([6.25])  # ------------------ What is (are) the value(s) of the reward(s)
        params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
        params[
            'episode_length'] = 40  # ---------------------------- 32 steps should be enough to reach the distal reward
        # params['use_epochs'] = False  # ------------------------- If [True] we use epochs, if [False] we use steps
    elif params['maze_type'] == 'Simple':
        params['start_pos'] = 6  # ----------------------------- What state do we start from
        params['rew_loc'] = np.array([2])  # --------------- What is (are) the rewarded state(s)
        params['rew_val'] = np.array([1])  # ----------------- What is (are) the value(s) of the reward(s)
        params['rew_prob'] = np.array([1])  # ---------------- What is (area) the probability/ies of the reward(s)
        params['new_rew_loc'] = np.array([0])  # ------------------ What is (are) the rewarded state(s)
        params['new_rew_val'] = np.array([1])  # ------------------ What is (are) the value(s) of the reward(s)
        params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
    elif params['maze_type'] == 'Linear':
        if not flickering:
            params['start_pos'] = 15  # ----------------------------- What state do we start from
            params['rew_loc'] = np.array([10, 0])  # --------------- What is (are) the rewarded state(s)
            params['rew_val'] = np.array([0.5, 5])  # ----------------- What is (are) the value(s) of the reward(s)
            params['rew_prob'] = np.array([1, 1])  # ---------------- What is (area) the probability/ies of the reward(s)
            params['new_rew_loc'] = None  # ------------------ What is (are) the rewarded state(s)
            params['new_rew_val'] = None  # ------------------ What is (are) the value(s) of the reward(s)
            params['new_rew_prob'] = None  # -- What is (area) the probability/ies of the reward(s)
            params['episode_length'] = 25  # ------------------------  Proximal rr: 10; distal; 50
        else:
            params['start_pos'] = 10  # ----------------------------- What state do we start from
            params['rew_loc'] = np.array([0, 15])  # --------------- What is (are) the rewarded state(s)
            params['rew_val'] = np.array([4, 3])  # ----------------- What is (are) the value(s) of the reward(s)
            params['rew_prob'] = np.array(
                [1, 0.5])  # ---------------- What is (area) the probability/ies of the reward(s)
            params['new_rew_loc'] = None  # ------------------ What is (are) the rewarded state(s)
            params['new_rew_val'] = None  # ------------------ What is (are) the value(s) of the reward(s)
            params['new_rew_prob'] = None  # -- What is (area) the probability/ies of the reward(s)
            params['episode_length'] = 25  # ------------------------  Proximal rr: 60*0.5; distal; 60*1
    elif params['maze_type'] == 'SED':
        params['start_pos'] = 3  # ----------------------------- What state do we start from
        params['rew_loc'] = np.array([22, 34])  # --------------- What is (are) the rewarded state(s)
        params['rew_val'] = np.array([0.5, 3])  # ----------------- What is (are) the value(s) of the reward(s)
        params['rew_prob'] = np.array([1, 1])  # ---------------- What is (area) the probability/ies of the reward(s)
        params['new_rew_loc'] = np.array([27])  # ------------------ What is (are) the rewarded state(s)
        params['new_rew_val'] = np.array([3])  # ------------------ What is (are) the value(s) of the reward(s)
        params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
        params['episode_length'] = 30  # ------------------------  Proximal rr: 10; distal; 50
    elif params['maze_type'] == 'DTwide':
        params['start_pos'] = 78  # ----------------------------- What state do we start from
        params['rew_loc'] = np.array([76, 36])  # --------------- What is (are) the rewarded state(s)
        params['rew_val'] = np.array([0.5, 3.5])  # ----------------- What is (are) the value(s) of the reward(s)
        params['rew_prob'] = np.array([1, 1])  # ---------------- What is (area) the probability/ies of the reward(s)
        params['new_rew_loc'] = np.array([32])  # ------------------ What is (are) the rewarded state(s)
        params['new_rew_val'] = np.array([3.5])  # ------------------ What is (are) the value(s) of the reward(s)
        params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
        params['episode_length'] = 45  # ------------------------  Proximal rr: 10; distal; 50
        # params['start_pos'] = 184  # ----------------------------- What state do we start from
        # params['rew_loc'] = np.array([181, 88])  # --------------- What is (are) the rewarded state(s)
        # params['rew_val'] = np.array([0.3, 3])  # ----------------- What is (are) the value(s) of the reward(s)
        # params['rew_prob'] = np.array([1, 1])  # ---------------- What is (area) the probability/ies of the reward(s)
        # params['new_rew_loc'] = np.array([82])  # ------------------ What is (are) the rewarded state(s)
        # params['new_rew_val'] = np.array([3])  # ------------------ What is (are) the value(s) of the reward(s)
        # params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
        # params['episode_length'] = 45  # ------------------------  Proximal rr: 10; distal; 50
    elif params['maze_type'] == 'DTtight':
        params['start_pos'] = 6  # ----------------------------- What state do we start from
        params['restricted_dT'] = restricted  # ---------------------- Is the movement restricted to unidirectional?
        if not (params['restricted_dT']):
            raise ValueError('Only coded for restricted version')
        params[
            'episode_length'] = None  # --------------------------- 10 steps should be enough to reach the distal reward
        params['rew_loc'] = np.array([5])  # --------------- What is (are) the rewarded state(s)
        params['rew_val'] = np.array([1])  # ----------------- What is (are) the value(s) of the reward(s)
        params['rew_prob'] = np.array([1])  # ---------------- What is (area) the probability/ies of the reward(s)
        params['new_rew_loc'] = np.array([7])  # ------------------ What is (are) the rewarded state(s)
        params['new_rew_val'] = np.array([1])  # ------------------ What is (are) the value(s) of the reward(s)
        params['new_rew_prob'] = np.array([1])  # -- What is (area) the probability/ies of the reward(s)
    elif params['maze_type'] == 'Tolman':
        params['start_pos'] = 20  # ----------------------------- What state do we start from
        params['rew_loc'] = np.array([0])  # --------------- What is (are) the rewarded state(s)
        params['rew_val'] = np.array([1])  # ----------------- What is (are) the value(s) of the reward(s)
        params['rew_prob'] = np.array([1])  # ---------------- What is (area) the probability/ies of the reward(s)
        params['new_rew_loc'] = None  # ------------------ What is (are) the rewarded state(s)
        params['new_rew_val'] = None  # ------------------ What is (are) the value(s) of the reward(s)
        params['new_rew_prob'] = None  # -- What is (area) the probability/ies of the reward(s)
        params[
            'episode_length'] = 22  # ---------------------------- rr: 10, 12 or 16

    params['num_runs'] = num_runs  # ---------------------------- How many epochs do we model
    if rew_change is None:
        params['rew_change'] = rew_change
    else:
        params['rew_change'] = math.ceil(
            params['num_runs'] * rew_change)  # - When do we change the reward location (if we do)
    params['env_forbidden_walls'] = forbidden_walls  # ----------------- Is it forbidden to bump into walls?
    params['slip_prob'] = 0  # ------------------------------ The probability of slipping after a step
    params['wall_loc'] = wall_loc  # ----------- The wall is between what states (before the change)
    if wall_change is None:
        params['wall_change'] = wall_change
    else:
        params['wall_change'] = math.ceil(
                params['num_runs'] * wall_change)   # ------- When do we add a wall (if we do)
    params['new_wall_loc'] = new_wall_loc  # ------------ The wall is between what states (after the change)

    # About the agent
    params[
        'known_env'] = known_env  # --------------------------- Is the state-space known in advance [True] or not [False]
    params['model'] = model_type  # ------------------------------- Model free or model based

    if params['model'] == 'MF':
        params['model_type'] = 'TD'  # ---------------------- TD (for MF) or VI/PI (for MB)
        params['alpha'] = 0.8  # ---------------------------- from Massi et al. (2022) MF-priority
    elif params['model'] == 'MB':
        params['model_type'] = 'VI'  # ---------------------- TD (for MF) or VI/PI (for MB)
        params['pre_training'] = None  # -------------------- how many steps do we pre-train
    params['gamma'] = g  # -------------------------------- Discounting factor
    params['nV'] = nV  # -------------------------------- The model is updated based on the last nV visits
    params['decision_rule'] = decision_rule  # -------------- Greedy decisions (could be 'max', 'softmax', 'epsilon')
    if params['decision_rule'] == 'epsilon':
        params['epsilon'] = epsilon  # -------------------------- Epsilon of the epsilon-greedy
    elif params['decision_rule'] == 'softmax':
        params['beta'] = beta  # ------------------------------ Beta for softmax from Massi et al. (2022) MF-priority
    params['replay_type'] = replay_type  # ------------------ 'priority', 'trsam', 'bidir', 'backwards', 'forward'
    params['replay_every_step'] = False
    if params['replay_type'] in ['trsam', 'itrsam']:
        params['replay_every_step'] = True
    if params['replay_type'] in ['priority']:
        params['event_handle'] = handle  # -------------------------- What is each new memory compared to [s, sa, sas]
    params['event_content'] = handle  # ------------------------ What is not estimated from model [s, sa, sas, sasr]
    params['replay_thresh'] = replay_threshold  # ----------- Smallest surprise necessary to initiate replay
    params['max_replay'] = max_replay  # ---------------------------- Max replay steps per replay event
    params['add_predecessors'] = add_predecessors  # -------- When do I add state predecessors (None, act, rep or both)
    params[
        'replay_forbidden_walls'] = False  # ----------------- If we replay (simulate), is bumping into a wall forbidden?
    params['dec_weights'] = dec_weights  # ------------------ The weights used for decision-making [Q, Ur, Ut] float
    params['rep_weigths'] = rep_weights  # ------------------ The weights used for replay [Q, Ur, Ut] float
    params['epist_rew_type'] = epist_rew_type  # ------------- The type of epistemic reward ["diff" or "abs"]

    # shuffled_replay_test(**params)
    params['num_visits_training'] = num_visits_training
    params['state_of_interest'] = state_of_interest
    params['action_of_interest'] = action_of_interest
    if state_of_interest is None:
        run_dT(**params)
    else:
        tolman_replay_test(**params)


def shuffled_replay_test(rew_loc: np.ndarray, start_pos: int,
                         model: str, model_type: str, gamma: float, nV: float, decision_rule: str,
                         **kwargs):
    """
    Introduces the agent to all the states and actions shuffled randomly, and then records the first replay event
    :param rew_loc: where the OG reward will be placed
    :param start_pos: where the agent starts from
    :param model: 'MF' or 'MB'
    :param model_type: 'TD', 'VI' or 'PI'
    :param gamma: discount factor
    :param nV: The model is updated based on the last nV visits
    :param decision_rule: 'max', 'epsilon' or 'softmax'
    :param kwargs:
        Environment-related variables:
            env_forbidden_walls: can the agent choose to bump into a wall [bool]
            restricted_dT: is the movement unidirectional or not [bool]
            slip_prob: probability of slipping while moving [float]
            rew_val: value of reward [float array]
            rew_prob: proba of reward [float array]
            wall_loc: Between what states will we have walls [array of 2D arrays of ints]
        Agent-related variables:
            known_env: is the state-space previously known [True] or not [False, default]
            based on 'model':
                alpha: learning parameter for the MF agent [float]
            based on 'decision_rule':
                epsilon: exploration constant of epsilon greedy agent [float]
                beta: exploitation constant of softmax agent [float]
            replay_type: 'forward', 'backward', 'priority', 'trsam', 'bidir' or None:
                replay_every_step: do I replay after every step [True, default] or only after receiving a reward [False]
                event_handle: what should we compare a new event to when trying to estimate if we need to
                    overwrite an old memory or not: states ['s'], state-action ['sa'] or
                    state-action-new state ['sas']. Only needed if replay_type is "priority" or "bidir"
                event_content: what should we replay, states ['s'], state-action ['sa'],
                    state-action-new state ['sas'], or state-action-new state-reward ['sasr', default].
                replay_thresh: replay threshold [float]
                max_replay: max number of replay steps [int]
                add_predecessors: for priority and bidir, when do I add predecessors to the buffer ['act', 'rep',
                    'both', None]
                replay_forbidden_walls: is choosing a wall forbidden for replay [True] or not [False]
            epist_rew_type: do epistemic rewards come from absolute uncertainty ["abs"] or changes in unc ["diff"]
            dec_weight: weight of the different quality functions contributing to decisions [Q, Ur, Ut], float array
            rep_weight: weight of the different quality functions contributing to replay [Q, Ur, Ut], float array
        Storing-related variables:
            prog_bar: should I show progress bar?
            save_data: Should we save the data generated [True] or not [False, default]
            save_path: Where should we save [str] (default: current folder)
            save_tag: What tag should I add to the end of the filename [str, default: None]
    :return:
    """
    # Arguments for the environment
    maze_type = kwargs.get('maze_type', "DT")
    env_forbidden_walls = kwargs.get('env_forbidden_walls', False)
    restricted_dT = kwargs.get('restricted_dT', False)
    slip_prob = kwargs.get('slip_prob', 0)
    rew_val = kwargs.get('rew_val', np.ones(rew_loc.shape))
    rew_prob = kwargs.get('rew_prob', np.ones(rew_loc.shape))
    wall_loc = kwargs.get('wall_loc', np.array([[]]))

    # Arguments for the model
    known_env = kwargs.get('known_env', False)
    alpha, epsilon, beta, pre_training = None, None, None, None
    if model == 'MF':
        alpha = kwargs.get('alpha', None)
    if decision_rule == 'epsilon':
        epsilon = kwargs.get('epsilon', None)
    elif decision_rule == 'softmax':
        beta = kwargs.get('beta', None)
    replay_type = kwargs.get('replay_type', None)
    event_handle = kwargs.get('event_handle', None)
    event_content = kwargs.get('event_content', 'sasr')
    replay_thresh = kwargs.get('replay_thresh', None)
    max_replay = kwargs.get('max_replay', None)
    add_predecessors = kwargs.get('add_predecessors', None)
    replay_forbidden_walls = kwargs.get('replay_forbidden_walls', True)
    dec_weights = kwargs.get('dec_weights', np.array([0.75, 0.2, 0.05]))
    rep_weights = kwargs.get('rep_weights', dec_weights)
    epist_rew_type = kwargs.get('epist_rew_type', 'diff')

    # Arguments about saving
    prog_bar = kwargs.get('prog_bar', False)
    save_data = kwargs.get('save_data', False)
    save_path = kwargs.get('save_path', None)
    save_tag = kwargs.get('save_tag', None)
    save_format = kwargs.get('save_format', 'full')

    # 0) Creating the environment and the agent within
    dTm = None
    if maze_type == 'DT':
        dTm = DTMaze(forbidden_walls=env_forbidden_walls, restricted_dT=restricted_dT,
                     slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'M':
        dTm = Mmaze(forbidden_walls=env_forbidden_walls,
                    slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Open':
        dTm = OpenMaze(kwargs.get('x_dim', None), kwargs.get('y_dim', None),
                       forbidden_walls=env_forbidden_walls,
                       slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'ED':
        dTm = EDmaze(forbidden_walls=env_forbidden_walls,
                     slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Simple':
        dTm = SimpleMaze(forbidden_walls=env_forbidden_walls,
                         slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Linear':
        dTm = LinearMaze(forbidden_walls=env_forbidden_walls,
                         slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'SED':
        dTm = SmallEDmaze(forbidden_walls=env_forbidden_walls,
                          slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'DTwide':
        dTm = DTMazeWide(forbidden_walls=env_forbidden_walls,
                         slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'DTtight':
        dTm = DTMazeTight(forbidden_walls=env_forbidden_walls, restricted_dT=restricted_dT,
                          slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Tolman':
        dTm = TolmanMaze(forbidden_walls=env_forbidden_walls,
                          slip_prob=slip_prob, start_pos=start_pos)

    agent = RLagent(dTm, model_type, gamma, nV, decision_rule,
                    alpha=alpha, beta=beta, epsilon=epsilon, known_env=known_env,
                    replay_type=replay_type, event_content=event_content, event_handle=event_handle,
                    replay_thresh=replay_thresh, max_replay=max_replay,
                    epist_rew_type=epist_rew_type, dec_weights=dec_weights, rep_weights=rep_weights,
                    add_predecessors=add_predecessors, forbidden_walls=replay_forbidden_walls, format=save_format)

    # 0) Placing down the walls
    if wall_loc is not None:
        for w_idx in range(wall_loc.shape[0]):
            dTm.place_wall(wall_loc[w_idx, 0], wall_loc[w_idx, 1])

    # 2) Preparing the experiment
    for r_idx in range(len(rew_loc)):
        dTm.place_reward(rew_loc[r_idx], rew_val[r_idx], rew_prob[r_idx])
    if save_data:
        dTm.toggle_save()
        agent.toggle_save()

    # 3) Learning every single state and action in a randomized order
    all_states = np.array(range(dTm.state_num()))
    np.random.shuffle(all_states)
    if prog_bar:
        print('Learning all states and actions...')
        pbar = tqdm(total=len(all_states))
    for s in all_states:
        dTm.place_agent(s)
        a_poss = dTm.possible_moves(s)
        np.random.shuffle(a_poss)
        for a in a_poss:
            s_prime, r = dTm.step(s, a)
            hr, ht = agent.model_learning(s, a, s_prime, r)
            agent.inference(s, a, s_prime, np.array([r, hr, ht]))
        if prog_bar:
            pbar.update(1)
    agent.memory_replay()

    if save_data:
        dTm.dump_env(path=save_path, label=save_tag)
        agent.dump_agent(path=save_path, label=save_tag)

def tolman_replay_test(rew_loc: np.ndarray, start_pos: int,
                         model: str, model_type: str, gamma: float, nV: float, decision_rule: str,
                         **kwargs):
    """
    Introduces the agent to all the states and actions shuffled randomly, and then records the first replay event
    :param rew_loc: where the OG reward will be placed
    :param start_pos: where the agent starts from
    :param model: 'MF' or 'MB'
    :param model_type: 'TD', 'VI' or 'PI'
    :param gamma: discount factor
    :param nV: The model is updated based on the last nV visits
    :param decision_rule: 'max', 'epsilon' or 'softmax'
    :param kwargs:
        Environment-related variables:
            env_forbidden_walls: can the agent choose to bump into a wall [bool]
            restricted_dT: is the movement unidirectional or not [bool]
            slip_prob: probability of slipping while moving [float]
            rew_val: value of reward [float array]
            rew_prob: proba of reward [float array]
            wall_loc: Between what states will we have walls [array of 2D arrays of ints]
        Agent-related variables:
            known_env: is the state-space previously known [True] or not [False, default]
            based on 'model':
                alpha: learning parameter for the MF agent [float]
            based on 'decision_rule':
                epsilon: exploration constant of epsilon greedy agent [float]
                beta: exploitation constant of softmax agent [float]
            replay_type: 'forward', 'backward', 'priority', 'trsam', 'bidir' or None:
                replay_every_step: do I replay after every step [True, default] or only after receiving a reward [False]
                event_handle: what should we compare a new event to when trying to estimate if we need to
                    overwrite an old memory or not: states ['s'], state-action ['sa'] or
                    state-action-new state ['sas']. Only needed if replay_type is "priority" or "bidir"
                event_content: what should we replay, states ['s'], state-action ['sa'],
                    state-action-new state ['sas'], or state-action-new state-reward ['sasr', default].
                replay_thresh: replay threshold [float]
                max_replay: max number of replay steps [int]
                add_predecessors: for priority and bidir, when do I add predecessors to the buffer ['act', 'rep',
                    'both', None]
                replay_forbidden_walls: is choosing a wall forbidden for replay [True] or not [False]
            epist_rew_type: do epistemic rewards come from absolute uncertainty ["abs"] or changes in unc ["diff"]
            dec_weight: weight of the different quality functions contributing to decisions [Q, Ur, Ut], float array
            rep_weight: weight of the different quality functions contributing to replay [Q, Ur, Ut], float array
        Storing-related variables:
            prog_bar: should I show progress bar?
            save_data: Should we save the data generated [True] or not [False, default]
            save_path: Where should we save [str] (default: current folder)
            save_tag: What tag should I add to the end of the filename [str, default: None]
    :return:
    """
    # Arguments for the environment
    maze_type = kwargs.get('maze_type', "DT")
    env_forbidden_walls = kwargs.get('env_forbidden_walls', False)
    restricted_dT = kwargs.get('restricted_dT', False)
    slip_prob = kwargs.get('slip_prob', 0)
    rew_val = kwargs.get('rew_val', np.ones(rew_loc.shape))
    rew_prob = kwargs.get('rew_prob', np.ones(rew_loc.shape))
    wall_loc = kwargs.get('wall_loc', np.array([[]]))
    wall_change = kwargs.get('wall_change', None)
    new_wall_loc = kwargs.get('new_wall_loc', np.array([[]]))
    num_visits_training = kwargs.get('num_visits_training', None)
    state_of_interest = kwargs.get('state_of_interest', None)
    action_of_interest = kwargs.get('action_of_interest', None)
    if state_of_interest is None or action_of_interest is None:
        raise ValueError('For the single replay scenario it is essential that a state_of_interest and action_of_interest be specified.')
    num_runs = kwargs.get('num_runs', 1)

    # Arguments for the model
    known_env = kwargs.get('known_env', False)
    alpha, epsilon, beta, pre_training = None, None, None, None
    if model == 'MF':
        alpha = kwargs.get('alpha', None)
    if decision_rule == 'epsilon':
        epsilon = kwargs.get('epsilon', None)
    elif decision_rule == 'softmax':
        beta = kwargs.get('beta', None)
    replay_type = kwargs.get('replay_type', None)
    event_handle = kwargs.get('event_handle', None)
    event_content = kwargs.get('event_content', 'sasr')
    replay_thresh = kwargs.get('replay_thresh', None)
    max_replay = kwargs.get('max_replay', None)
    add_predecessors = kwargs.get('add_predecessors', None)
    replay_forbidden_walls = kwargs.get('replay_forbidden_walls', True)
    dec_weights = kwargs.get('dec_weights', np.array([0.75, 0.2, 0.05]))
    rep_weights = kwargs.get('rep_weights', dec_weights)
    epist_rew_type = kwargs.get('epist_rew_type', 'diff')

    # Arguments about saving
    prog_bar = kwargs.get('prog_bar', False)
    save_data = kwargs.get('save_data', False)
    save_path = kwargs.get('save_path', None)
    save_tag = kwargs.get('save_tag', None)
    save_format = kwargs.get('save_format', 'full')

    # 0) Creating the environment and the agent within
    dTm = None
    if maze_type == 'DT':
        dTm = DTMaze(forbidden_walls=env_forbidden_walls, restricted_dT=restricted_dT,
                     slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'M':
        dTm = Mmaze(forbidden_walls=env_forbidden_walls,
                    slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Open':
        dTm = OpenMaze(kwargs.get('x_dim', None), kwargs.get('y_dim', None),
                       forbidden_walls=env_forbidden_walls,
                       slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'ED':
        dTm = EDmaze(forbidden_walls=env_forbidden_walls,
                     slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Simple':
        dTm = SimpleMaze(forbidden_walls=env_forbidden_walls,
                         slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Linear':
        dTm = LinearMaze(forbidden_walls=env_forbidden_walls,
                         slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'SED':
        dTm = SmallEDmaze(forbidden_walls=env_forbidden_walls,
                          slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'DTwide':
        dTm = DTMazeWide(forbidden_walls=env_forbidden_walls,
                         slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'DTtight':
        dTm = DTMazeTight(forbidden_walls=env_forbidden_walls, restricted_dT=restricted_dT,
                          slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Tolman':
        dTm = TolmanMaze(forbidden_walls=env_forbidden_walls,
                          slip_prob=slip_prob, start_pos=start_pos)

    agent = RLagent(dTm, model_type, gamma, nV, decision_rule,
                    alpha=alpha, beta=beta, epsilon=epsilon, known_env=known_env,
                    replay_type=replay_type, event_content=event_content, event_handle=event_handle,
                    replay_thresh=replay_thresh, max_replay=max_replay,
                    epist_rew_type=epist_rew_type, dec_weights=dec_weights, rep_weights=rep_weights,
                    add_predecessors=add_predecessors, forbidden_walls=replay_forbidden_walls,
                    format=save_format)

    # 0) Placing down the walls
    if wall_loc is not None:
        for w_idx in range(wall_loc.shape[0]):
            dTm.place_wall(wall_loc[w_idx, 0], wall_loc[w_idx, 1])

    # 2) Preparing the experiment
    for r_idx in range(len(rew_loc)):
        dTm.place_reward(rew_loc[r_idx], rew_val[r_idx], rew_prob[r_idx])

    # 3) Learning every single state and action in a randomized order
    if num_visits_training is None:
        num_visits_training = 1
    all_states = np.array(range(dTm.state_num()))
    if prog_bar:
        print('Learning all states and actions...')
        pbar = tqdm(total=num_visits_training)
    for _ in range(num_visits_training):
        np.random.shuffle(all_states)
        for s in all_states:
            dTm.place_agent(s)
            a_poss = dTm.possible_moves(s)
            np.random.shuffle(a_poss)
            for a in a_poss:
                s_prime, r = dTm.step(s, a)
                hr, ht = agent.model_learning(s, a, s_prime, r)
                delta_C = agent.inference(s, a, s_prime, np.array([r, hr, ht]))
                if replay_thresh is not None and abs(delta_C) > replay_thresh:
                    agent.memory_replay(s=s)
        if prog_bar:
            pbar.update(1)

    # 4) Placing down the new walls
    if wall_change is not None:
        dTm.reset_wall()
        for w_idx in range(new_wall_loc.shape[0]):
            dTm.place_wall(new_wall_loc[w_idx, 0], new_wall_loc[w_idx, 1])

    # 5) The actual replay event
    if save_data:
        dTm.toggle_save()
        agent.toggle_save()

    for _ in range(num_runs):
        dTm.reset_agent()  # This indicates to the environment that we're starting a new trial
        dTm.place_agent(state_of_interest)
        a_poss = dTm.possible_moves(state_of_interest)
        a = agent.choose_action(state_of_interest, a_poss)
        a = action_of_interest
        s_prime, r = dTm.step(state_of_interest, a)
        hr, ht = agent.model_learning(state_of_interest, a, s_prime, r)  # The epistemic rewards
        delta_C = agent.inference(state_of_interest, a, s_prime, np.array([r, hr, ht]))
        if replay_thresh is not None and abs(delta_C) > replay_thresh:
            agent.memory_replay(s=state_of_interest)

    if save_data:
        dTm.dump_env(path=save_path, label=save_tag)
        agent.dump_agent(path=save_path, label=save_tag)


def run_dT(rew_loc: np.ndarray, start_pos: int, episode_length: int, num_runs: int,
           model: str, model_type: str, gamma: float, nV: float, decision_rule: str,
           **kwargs):
    """
    Runs the double-T-maze experiment
    :param rew_loc: where the OG reward will be placed
    :param start_pos: where the agent starts from
    :param episode_length: the length oif each epoch in steps
    :param num_runs: how many steps/epochs are we modelling
    :param model: 'MF' or 'MB'
    :param model_type: 'TD', 'VI' or 'PI'
    :param gamma: discount factor
    :param nV: The model is updated based on the last nV visits
    :param decision_rule: 'max', 'epsilon' or 'softmax'
    :param kwargs:
        Environment-related variables:
            teleport: if True, the agent teleports back to its initial position upon receiving a reward
            use_epochs: if [True] we use epochs instead of steps (an epoch ends when a reward is received)
            env_forbidden_walls: can the agent choose to bump into a wall [bool]
            restricted_dT: is the movement unidirectional or not [bool]
            slip_prob: probability of slipping while moving [float]
            rew_val: value of reward [float array]
            rew_prob: proba of reward [float array]
            rew_change: what step will we change the reward location (if we do) [int]
                new_rew_loc: where the reward will be placed after the location change [int array]
                new_rew_val: value of reward [float array]
                new_rew_prob: proba of reward [float array]
            wall_loc: Between what states will we have walls [array of 2D arrays of ints]
            wall_change: what step will we add new walls (if we do) [int]
                new_wall_loc: Between what states will the wall(s) lie
        Agent-related variables:
            known_env: is the state-space previously known [True] or not [False, default]
            based on 'model':
                alpha: learning parameter for the MF agent [float]
                pre_training: number of unrewarded pre-training steps (for tuning the model of MB agent) [int]
            based on 'decision_rule':
                epsilon: exploration constant of epsilon greedy agent [float]
                beta: exploitation constant of softmax agent [float]
            replay_type: 'forward', 'backward', 'priority', 'trsam', 'bidir' or None:
                replay_every_step: do I replay after every step [True, default] or only after receiving a reward [False]
                event_handle: what should we compare a new event to when trying to estimate if we need to
                    overwrite an old memory or not: states ['s'], state-action ['sa'] or
                    state-action-new state ['sas']. Only needed if replay_type is "priority" or "bidir"
                event_content: what should we replay, states ['s'], state-action ['sa'],
                    state-action-new state ['sas'], or state-action-new state-reward ['sasr', default].
                replay_thresh: replay threshold [float]
                max_replay: max number of replay steps [int]
                add_predecessors: for priority and bidir, when do I add predecessors to the buffer ['act', 'rep',
                    'both', None]
                replay_forbidden_walls: is choosing a wall forbidden for replay [True] or not [False]
            epist_rew_type: do epistemic rewards come from absolute uncertainty ["abs"] or changes in unc ["diff"]
            dec_weight: weight of the different quality functions contributing to decisions [Q, Ur, Ut], float array
            rep_weight: weight of the different quality functions contributing to replay [Q, Ur, Ut], float array
        Storing-related variables:
            prog_bar: should I show progress bar?
            save_data: Should we save the data generated [True] or not [False, default]
            save_path: Where should we save [str] (default: current folder)
            save_tag: What tag should I add to the end of the filename [str, default: None]
    :return:
    """
    # Arguments for the environment
    maze_type = kwargs.get('maze_type', "DT")
    teleport = kwargs.get('teleport', True)
    env_forbidden_walls = kwargs.get('env_forbidden_walls', False)
    restricted_dT = kwargs.get('restricted_dT', False)
    slip_prob = kwargs.get('slip_prob', 0)
    rew_val = kwargs.get('rew_val', np.ones(rew_loc.shape))
    rew_prob = kwargs.get('rew_prob', np.ones(rew_loc.shape))
    rew_change = kwargs.get('rew_change', None)
    new_rew_loc = kwargs.get('new_rew_loc', rew_loc)
    new_rew_val = kwargs.get('new_rew_val', rew_val)
    new_rew_prob = kwargs.get('new_rew_prob', rew_prob)
    wall_loc = kwargs.get('wall_loc', np.array([[]]))
    wall_change = kwargs.get('wall_change', None)
    new_wall_loc = kwargs.get('new_wall_loc', np.array([[]]))

    # Arguments for the model
    known_env = kwargs.get('known_env', False)
    alpha, epsilon, beta, pre_training = None, None, None, None
    if model == 'MF':
        alpha = kwargs.get('alpha', None)
    elif model == 'MB':
        pre_training = kwargs.get('pre_training', None)
    if decision_rule == 'epsilon':
        epsilon = kwargs.get('epsilon', None)
    elif decision_rule == 'softmax':
        beta = kwargs.get('beta', None)
    replay_type = kwargs.get('replay_type', None)
    event_handle = kwargs.get('event_handle', None)
    event_content = kwargs.get('event_content', 'sasr')
    replay_every_step = kwargs.get('replay_every_step', True)
    replay_thresh = kwargs.get('replay_thresh', None)
    max_replay = kwargs.get('max_replay', None)
    add_predecessors = kwargs.get('add_predecessors', None)
    replay_forbidden_walls = kwargs.get('replay_forbidden_walls', True)
    epist_rew_type = kwargs.get('epist_rew_type', 'diff')
    dec_weights = kwargs.get('dec_weights', np.array([0.75, 0.2, 0.05]))
    rep_weights = kwargs.get('rep_weights', dec_weights)

    # Arguments about saving
    prog_bar = kwargs.get('prog_bar', False)
    save_data = kwargs.get('save_data', False)
    save_path = kwargs.get('save_path', None)
    save_tag = kwargs.get('save_tag', None)
    save_format = kwargs.get('save_format', 'full')

    # 0) Creating the environment and the agent within
    dTm = None
    if maze_type == 'DT':
        dTm = DTMaze(forbidden_walls=env_forbidden_walls, restricted_dT=restricted_dT,
                     slip_prob=slip_prob, teleport=teleport, start_pos=start_pos)
    elif maze_type == 'M':
        dTm = Mmaze(forbidden_walls=env_forbidden_walls,
                    slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Open':
        dTm = OpenMaze(kwargs.get('x_dim', None), kwargs.get('y_dim', None),
                       forbidden_walls=env_forbidden_walls,
                       slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'ED':
        dTm = EDmaze(forbidden_walls=env_forbidden_walls,
                     slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Simple':
        dTm = SimpleMaze(forbidden_walls=env_forbidden_walls,
                         slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Linear':
        dTm = LinearMaze(forbidden_walls=env_forbidden_walls,
                         slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'SED':
        dTm = SmallEDmaze(forbidden_walls=env_forbidden_walls,
                          slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'DTwide':
        dTm = DTMazeWide(forbidden_walls=env_forbidden_walls,
                         slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'DTtight':
        dTm = DTMazeTight(forbidden_walls=env_forbidden_walls, restricted_dT=restricted_dT,
                          slip_prob=slip_prob, start_pos=start_pos)
    elif maze_type == 'Tolman':
        dTm = TolmanMaze(forbidden_walls=env_forbidden_walls,
                          slip_prob=slip_prob, start_pos=start_pos)

    agent = RLagent(dTm, model_type, gamma, nV, decision_rule,
                    alpha=alpha, beta=beta, epsilon=epsilon, known_env=known_env,
                    replay_type=replay_type, event_content=event_content, event_handle=event_handle,
                    replay_thresh=replay_thresh, max_replay=max_replay,
                    epist_rew_type=epist_rew_type, dec_weights=dec_weights, rep_weights=rep_weights,
                    add_predecessors=add_predecessors, forbidden_walls=replay_forbidden_walls, format=save_format)

    # print(f'Before execution:\n\t- agent: {asizeof(agent)} bytes\n\t- env: {asizeof(dTm)} bytes')

    # 0) Placing down the walls
    if wall_loc is not None:
        for w_idx in range(wall_loc.shape[0]):
            dTm.place_wall(wall_loc[w_idx, 0], wall_loc[w_idx, 1])

    if model == 'MB':
        # 1) Pre-training if the agent is MB
        run_experiment(pre_training, episode_length, dTm, agent, pre_training=True, restricted=restricted_dT)
        dTm.place_agent(start_pos)

    # 2) Preparing the experiment
    for r_idx in range(len(rew_loc)):
        dTm.place_reward(rew_loc[r_idx], rew_val[r_idx], rew_prob[r_idx])

    ########################################################################
    # dTm.print_map('restr1')
    # for r_idx in range(len(new_rew_loc)):
    #     dTm.place_reward(new_rew_loc[r_idx], new_rew_val[r_idx], new_rew_prob[r_idx])
    # dTm.print_map('restr3')
    # dTm.reset_reward()
    # for r_idx in range(len(new_rew_loc)):
    #     dTm.place_reward(new_rew_loc[r_idx], new_rew_val[r_idx], new_rew_prob[r_idx])
    # dTm.print_map('restr2')
    ########################################################################

    if save_data:
        dTm.toggle_save()
        agent.toggle_save()

    # 3) Running the experiment (with and without reward change)
    if rew_change is None and wall_change is None:
        run_experiment(num_runs, episode_length, dTm, agent, replay_thresh=replay_thresh,
                       replay_every_step=replay_every_step, prog_bar=prog_bar, restricted=restricted_dT)
    else:
        first_stop = min([x for x in [rew_change, wall_change] if x is not None])
        last_stop = max([x for x in [rew_change, wall_change] if x is not None])
        if last_stop == first_stop:  # Only 1 stop
            last_stop = num_runs
        run_experiment(first_stop, episode_length, dTm, agent, replay_thresh=replay_thresh,
                       replay_every_step=replay_every_step, prog_bar=prog_bar, restricted=restricted_dT)
        if first_stop == rew_change:
            dTm.reset_reward()
            for r_idx in range(len(new_rew_loc)):
                dTm.place_reward(new_rew_loc[r_idx], new_rew_val[r_idx], new_rew_prob[r_idx])
        else:
            dTm.reset_wall()
            for w_idx in range(new_wall_loc.shape[0]):
                dTm.place_wall(new_wall_loc[w_idx, 0], new_wall_loc[w_idx, 1])
        run_experiment(last_stop - first_stop, episode_length, dTm, agent, replay_thresh=replay_thresh,
                       replay_every_step=replay_every_step, prog_bar=prog_bar, restricted=restricted_dT)
        if last_stop != num_runs:  # We have an actual second stop
            if last_stop == rew_change:
                dTm.reset_reward()
                for r_idx in range(len(new_rew_loc)):
                    dTm.place_reward(new_rew_loc[r_idx], new_rew_val[r_idx], new_rew_prob[r_idx])
            else:
                dTm.reset_wall()
                for w_idx in range(new_wall_loc.shape[0]):
                    dTm.place_wall(new_wall_loc[w_idx, 0], new_wall_loc[w_idx, 1])
            run_experiment(num_runs - last_stop, episode_length, dTm, agent, replay_thresh=replay_thresh,
                           replay_every_step=replay_every_step, prog_bar=prog_bar, restricted=restricted_dT)

    # 4) Save everything
    # print(f'After execution:\n\t- agent: {asizeof(agent)} bytes\n\t- env: {asizeof(dTm)} bytes')

    if save_data:
        # Let's put back all the rewards and walls for plotting purposes
        if rew_change is not None:
            for r_idx in range(len(rew_loc)):
                dTm.place_reward(rew_loc[r_idx], rew_val[r_idx], rew_prob[r_idx])
        if wall_loc is not None:
            for w_idx in range(wall_loc.shape[0]):
                dTm.place_wall(wall_loc[w_idx, 0], wall_loc[w_idx, 1])
        dTm.dump_env(path=save_path, label=save_tag)
        agent.dump_agent(path=save_path, label=save_tag)


def run_experiment(num_runs: int, episode_length, env: Env, agent: RLagent, **kwargs):
    """
    Runs an experiment of a pre-defined length
    :param num_runs: how many epochs we are modelling
    :param episode_length: how long each episode is
    :param env: what is the environment
    :param agent: what is the agent
    :param kwargs:
        prog_bar: do I show a progress bar
        replay_thresh: what is the threshold to trigger replay (if None, no replay)
        pre_training: is this a pre-training setting [True -- no need to learn Q values] or not [False]
        replay_every_step: do I replay after each step [True -- default] or only after a significant event [False]
        restricted: is it a restricted DT-maze [True] or not [False, default]
    :return:
    """
    if num_runs is None:
        return
    restricted = kwargs.get('restricted', False)
    prog_bar = kwargs.get('prog_bar', False)
    replay_thresh = kwargs.get('replay_thresh', None)
    pre_training = kwargs.get('pre_training', False)
    replay_every_step = kwargs.get('replay_every_step', True)
    if prog_bar:
        pbar = tqdm(total=num_runs)
    for episode in range(num_runs):
        # 0) put back the agent to where it belongs
        env.reset_agent()

        step = 0
        s_start = env.curr_state()
        s_prime = None
        while (episode_length is not None and step < episode_length) or (episode_length is None and s_prime != s_start):
            # If fixed episoe length we simply loop throught he steps. Otherwise we wait for returning to the starting state
            # 1) Observe the environment
            s = env.curr_state()
            a_poss = env.possible_moves(s)

            # 2) Choose an action
            a = agent.choose_action(s, a_poss)

            # 3) Perform a step
            s_prime, r = env.step(s, a)

            # 4) Learn
            hr, ht = agent.model_learning(s, a, s_prime, r)  # The epistemic rewards
            if not pre_training:
                # print(f"\n\nStep taken: s={s}, a={a}, s'={s_prime}, r={r}")  ###########
                delta_C = agent.inference(s, a, s_prime, np.array([r, hr, ht]))
                if replay_every_step or (replay_thresh is not None and abs(delta_C) > replay_thresh):
                    agent.memory_replay(s=s_prime)

            step += 1

        if prog_bar:
            pbar.update(1)
