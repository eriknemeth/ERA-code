import matplotlib.pyplot as plt

from classes.env import *
import scipy
# from statannot import add_stat_annotation
from statannotations.Annotator import Annotator
import matplotlib
from scipy.stats import linregress
from collections import deque
class PlotterEnv(Env):
    """
    A child class of Env that can load in the event history of an agent and plot it
    """

    def __init__(self, file_name: str, **kwargs):
        """
        This class will only ever be used to plot previous results, thus we can only call it by loading a file
        :param file_name: the name of the environment file [csv]
        :param kwargs:
            path: path to the file. If nothing is specified we'll be looking in the working folder
            norm_rep: None -- no normalization (counts);
                      'total' -- normalize replay by total number of replay steps and stops by total stops
                      'visits' -- normalize replay by total number of replay steps and stops by all visits to the state
            params: what parameters were used to create the batches. If not None, then cumulative reward rates will be
                stored that can be plotted as a heatmap-like matrix
                win_begin: if we compute the cumulative reward rate (matrix-representation), then where does the
                    interval begin, over which we accumulate the obtained reward [optional int (step/epoch number),
                    default=0]
                win_end: if we compute the cumulative reward rate (matrix-representation), then where does the
                    interval end, over which we accumulate the obtained reward [optional int (step/epoch number),
                    default=None, can be negative integer]
        :return:
        """
        Env.__init__(self)
        # Parameters of plotting
        self._norm_rep = kwargs.get('norm_rep', None)  # Will we normalize the replay maps based on total replay count?
        if self._norm_rep not in [None, 'total', 'visits']:
            raise ValueError('Nonmalization has to be either by total number of samples or by the number of visis.')
        # Placeholders for the agent data
        self._agent_events = None
        # The following variables are structured as {batch_name: np.ndarray(x, y, repetition), ...}
        self._crossed = {}
        self._stopped_at = {}  # A dict of arrays (of the maze) where each state counts stoppings by the agent
        self._replayed = {}  # A dict of arrays (of the maze) where each state counts replays by the agent
        # The following variables are structured as {batch_name: np.ndarray(t, repetition), ...}
        self._rew_rate = {}  # A dict of vectors containing the reward rate for each step/epoch
        self._replay_rate = {}  # A dict of vectors containing the replay rate for each step/epoch
        self._Ut_means = {}  # A dict of arrays, each array consisting of vectors containing the mean of the maximal
        # (legal) Ut val for each state
        self._Ur_means = {}  # Same but for Ur

        # In case of a need for matrix-like representation
        params = kwargs.get('params', None)
        if params is not None:
            params.append('cumul_rew')
            self._cumul_rew = pd.DataFrame(columns=params)
        self._win_begin = kwargs.get('win_begin', 0)
        self._win_end = kwargs.get('win_end', None)

        # The environment itself
        path = kwargs.get('path', None)
        self.load_env(file_name, path=path)
        return

    # Internal methods of the data aggreagtor
    def __count_replay__(self, batch: str) -> None:
        """
        Counts the replay events (stoppings) and the replayed states and stores them
        :param batch: the name of the batch of data we're working on
        :return:
        """
        # First extend the current memory
        if batch not in self._stopped_at:
            # If it's a new batch, then let's just add a corresponding array
            self._replayed[batch] = np.zeros((self._maze.shape[0], self._maze.shape[1], 1))
            self._stopped_at[batch] = np.zeros((self._maze.shape[0], self._maze.shape[1], 1))
            self._crossed[batch] = np.zeros((self._maze.shape[0], self._maze.shape[1], 1))
        else:
            # If it is a pre-existing batch, let's simply extend it
            self._replayed[batch] = np.append(self._replayed[batch],
                                              np.zeros((self._maze.shape[0], self._maze.shape[1], 1)),
                                              axis=2)
            self._stopped_at[batch] = np.append(self._stopped_at[batch],
                                                np.zeros((self._maze.shape[0], self._maze.shape[1], 1)),
                                                axis=2)
            self._crossed[batch] = np.append(self._crossed[batch],
                                             np.zeros((self._maze.shape[0], self._maze.shape[1], 1)),
                                             axis=2)

        # Preparing the table to use, including the index by which we'll cut it up
        agent_events = self._agent_events.copy(deep=False)
        agent_events = self.__cut_window__(agent_events)

        # For potential normalization:
        # visited = None
        # if self._norm_rep == 'visits':
        #     visited = np.zeros((self._maze.shape[0], self._maze.shape[1]))

        # Loop through the whole table
        for row_idx in agent_events.index:
            if agent_events['step'][row_idx] > 0:  # If we have a replay event
                # We update the replayed states
                x, y = self.__find_state_coord__(row_idx, 's')
                self._replayed[batch][x, y, -1] += 1
                x, y = self.__find_state_coord__(row_idx, 's_prime')
                self._replayed[batch][x, y, -1] += 1

                if agent_events['step'][row_idx] == 1:  # And if this is really the first replay step
                    # We update stopping
                    x, y = self.__find_state_coord__(row_idx - 1, 's_prime')
                    self._stopped_at[batch][x, y, -1] += 1
            else:
                # This is a real step, so we have to count the visit for normalization purposes
                # if self._norm_rep == 'visits':
                x, y = self.__find_state_coord__(row_idx, 's_prime')
                self._crossed[batch][x, y, -1] += 1

        # Finally we normalize
        if self._norm_rep is not None:
            sum_of_steps = (agent_events['step'] > 0).to_numpy().sum()
            if sum_of_steps > 0:
                self._replayed[batch][:, :, -1] /= sum_of_steps
            if self._norm_rep == 'total':
                sum_of_replay = (agent_events['step'] == 1).to_numpy().sum()
                if sum_of_replay > 0:
                    self._stopped_at[batch][:, :, -1] /= sum_of_replay
            else:
                visited = np.copy(self._crossed[:, :, -1])
                visited[visited == 0] = 1
                self._stopped_at[batch][:, :, -1] = self._stopped_at[batch][:, :, -1] / visited
        return

    def __count_replay_in_chunks__(self, start_step: int, end_step: int):
        """
        Counts the replayed states within some pre-defined windows. This will only make sense if we run an experiment
        with a single large replay event, and we want to plot the evolution of the replay within a single batch of data.
        :param start_step: the number of the starting replay step
        :param end_step: the number of the ending replay step
        :return:
        """
        # First extend the current memory
        batch = start_step
        if batch not in self._stopped_at:
            # If it's a new batch, then let's just add a corresponding array
            self._replayed[batch] = np.zeros((self._maze.shape[0], self._maze.shape[1], 1))
            self._stopped_at[batch] = np.zeros((self._maze.shape[0], self._maze.shape[1], 1))
        else:
            # If it is a pre-existing batch, let's simply extend it
            self._replayed[batch] = np.append(self._replayed[batch],
                                              np.zeros((self._maze.shape[0], self._maze.shape[1], 1)),
                                              axis=2)
            self._stopped_at[batch] = np.append(self._stopped_at[batch],
                                                np.zeros((self._maze.shape[0], self._maze.shape[1], 1)),
                                                axis=2)

        for row_idx in range(len(self._agent_events)):
            if start_step < self._agent_events['step'].iloc[row_idx] < end_step:  # If we have a replay event
                # We update the replayed states
                x, y = self.__find_state_coord__(row_idx, 's')
                self._replayed[batch][x, y, -1] += 1
                x, y = self.__find_state_coord__(row_idx, 's_prime')
                self._replayed[batch][x, y, -1] += 1

        return

    def __epoch_based_rate_computer__(self, data: dict, batch: str, agent_events: pd.core.frame.DataFrame,
                                      **kwargs) -> None:
        """
        Computes the reward or the replay rates in an epoch-based fashion. The core of the computation is finding the
        sum of rewards or the sum of replay steps for each epoch
        :param data: either the self._rew_rate or the self._replay_rate
        :param batch: what batch we are working on
        :param agent_events: a dataframe with steps, iterations, rewards and replay
        :param kwargs:
            rate: if 'reward' (default) we compute reward rates. If 'replay' we compute replay rates
        :return:
        """
        rate = kwargs.get('rate', 'r')
        if rate == 'reward':
            rate = 'r'
        elif rate not in ['reward', 'r', 'replay']:
            raise ValueError('We can either compute reward or replay rates.')

        # If we consider epochs, then the reward rates is a fixed length vector
        # 1) Let's see if we have the requested batch
        batch_length = max(agent_events['ep']) - min(agent_events['ep']) + 1
        if batch not in data:
            # If it's a new batch, then let's just add a corresponding array
            # IMPORTANT: The zeroth episode will only last a single step (the setting up) and will be ignored
            data[batch] = np.zeros((batch_length, 1))
        else:
            # If it is a pre-existing batch, let's simply extend it
            data[batch] = np.append(data[batch],
                                    np.zeros((batch_length, 1)),
                                    axis=1)

        # 2) Loop through the dataframe and sum the reward/replay over each episode
        data[batch][:, -1] = agent_events.groupby(['ep'])[rate].sum()

        ################################################################################################################
        # plt.figure()
        # plt.plot(data[batch][:, -1])
        # plt.show()
        # plt.close()
        ################################################################################################################
        return

    # def __step_based_rate_computer__(self, data: dict, batch: str, agent_events: pd.core.frame.DataFrame,
    #                                  **kwargs) -> None:
    #     """
    #     Computes the reward or the replay rates in a step-based fashion. The core of the computation is
    #     a convolution over the reward or replay rate with a predefined smoothing window
    #     :param data: either the self._rew_rate or the self._replay_rate
    #     :param batch: what batch we are working on
    #     :param agent_events: a dataframe with steps, iterations, rewards and replay
    #     :param kwargs:
    #         rate: if 'reward' (default) we compute reward rates. If 'replay' we compute replay rates
    #         Ut_mean: a series containing the Ut means
    #         Ur_mean: a series containing the Ur means
    #     :return:
    #     """
    #     Ut_mean = kwargs.get('Ut_mean', None)
    #     Ur_mean = kwargs.get('Ur_mean', None)
    #     rate = kwargs.get('rate', 'r')
    #     if rate == 'reward':
    #         rate = 'r'
    #     elif rate not in ['reward', 'r', 'replay']:
    #         raise ValueError('We can either compute reward or replay rates.')
    #
    #     # If it is not measured in epochs, then we do a convolution
    #     conv_win = np.ones(self._win_size) * 1 / self._win_size
    #
    #     # 1) We compute the convolved reward rate.
    #     # We will pad the rewards on the left, but not on the right!
    #     og_rate = np.append(np.zeros(math.floor(self._win_size / 2)), agent_events[rate].to_numpy())
    #     smooth_rate = np.convolve(og_rate, conv_win, mode='valid')
    #     if Ur_mean is not None:
    #         Ur_mean = Ur_mean.to_numpy()
    #     if Ut_mean is not None:
    #         Ut_mean = Ut_mean.to_numpy()
    #
    #     # 2) Let's see if we have the requested batch
    #     if batch not in data:
    #         # If it's a new batch, then let's just add a corresponding array
    #         data[batch] = np.reshape(smooth_rate, (len(smooth_rate), 1))
    #         if Ur_mean is not None:
    #             self._Ur_means[batch] = np.reshape(Ur_mean, (len(Ur_mean), 1))
    #         if Ut_mean is not None:
    #             self._Ut_means[batch] = np.reshape(Ut_mean, (len(Ut_mean), 1))
    #     else:
    #         # If it is a pre-existing batch, we might need to pad it or the old ones on the right
    #         if data[batch].shape[0] > len(smooth_rate):
    #             smooth_rate = np.append(smooth_rate, smooth_rate[-1] * np.ones(data[batch].shape[0] - len(smooth_rate)))
    #             if Ur_mean is not None:
    #                 Ur_mean = np.append(Ur_mean,
    #                                     Ur_mean[-1] * np.ones(self._Ur_means[batch].shape[0] - len(Ur_mean)))
    #             if Ut_mean is not None:
    #                 Ut_mean = np.append(Ut_mean,
    #                                     Ut_mean[-1] * np.ones(self._Ut_means[batch].shape[0] - len(Ut_mean)))
    #         elif data[batch].shape[0] < len(smooth_rate):
    #             extension = np.repeat(data[batch][[-1], :], len(smooth_rate) - data[batch].shape[0], axis=0)
    #             data[batch] = np.append(data[batch], extension, axis=0)
    #             if Ur_mean is not None:
    #                 extension = np.repeat(self._Ur_means[batch][[-1], :], len(Ur_mean) - self._Ur_means[batch].shape[0],
    #                                       axis=0)
    #                 self._Ur_means[batch] = np.append(self._Ur_means[batch], extension, axis=0)
    #             if Ut_mean is not None:
    #                 extension = np.repeat(self._Ut_means[batch][[-1], :], len(Ut_mean) - self._Ut_means[batch].shape[0],
    #                                       axis=0)
    #                 self._Ut_means[batch] = np.append(self._Ut_means[batch], extension, axis=0)
    #
    #         # And then we add it to the end
    #         data[batch] = np.append(data[batch], np.reshape(smooth_rate, (smooth_rate.shape[0], 1)), axis=1)
    #         if Ur_mean is not None:
    #             self._Ur_means[batch] = np.append(self._Ur_means[batch], np.reshape(Ur_mean, (Ur_mean.shape[0], 1)),
    #                                               axis=1)
    #         if Ut_mean is not None:
    #             self._Ut_means[batch] = np.append(self._Ut_means[batch], np.reshape(Ut_mean, (Ut_mean.shape[0], 1)),
    #                                               axis=1)

    def __cut_window__(self, agent_events: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        """
        Cuts a window out of the given dataframe based on win_start and win_end
        :param agent_events: the adataframe to cut up
        :return: the window of interest
        """
        agent_events = agent_events.merge(self._events[['iter', 'ep']], on='iter', how='left')  # ep indexed from 1

        # 2) Preparing the indices of the cuts whether or not they are epoch or step based
        win_begin = self._win_begin  # Win begin can be positive or negative integer
        if win_begin < 0:
            win_begin = agent_events['ep'].iloc[-1] + win_begin
        elif win_begin == 0:
            win_begin = 1  # Episodes are actually numbered from 1
        win_end = self._win_end  # Win end can be None (till the end) positive (iter num) or negative (end-iter num)
        if win_end is None:
            win_end = agent_events['ep'].iloc[-1]
        elif win_end < 0:
            win_end = agent_events['ep'].iloc[-1] + win_end
        return agent_events.loc[(agent_events['ep'] >= win_begin) & (agent_events['ep'] < win_end), :]

    def __compute_U_dynamics(self, batch: str) -> None:
        """
        Computes the U-value dynamics
        :param batch: The name of the batch of data we're working on
        """
        # This is for the U-value dynamics
        # First we find the maximal U-value associated with each state over all legal actions
        agent_events = self._agent_events.copy(deep=True)
        agent_events = agent_events[agent_events['step'] == 0]
        # self._agent_events.reset_index(drop=True, inplace=True)
        agent_events = self.__cut_window__(agent_events)
        Ur_vals = self.__find_max_vals__(agent_events, 'Ur')
        Ut_vals = self.__find_max_vals__(agent_events, 'Ut')

        # Now we find the mean value amongst all states
        Ur_mean = Ur_vals.mean(axis=1)
        Ut_mean = Ut_vals.mean(axis=1)

        # 1) Let's see if we have the requested batch
        batch_length = max(agent_events['ep']) - min(agent_events['ep']) + 1
        if batch not in self._Ur_means:
            # If it's a new batch, then let's just add a corresponding array
            # IMPORTANT: The zeroth episode will only last a single step (the setting up) and will be ignored
            self._Ur_means[batch] = np.zeros((batch_length, 1))
            self._Ut_means[batch] = np.zeros((batch_length, 1))
        else:
            # If it is a pre-existing batch, let's simply extend it
            self._Ur_means[batch] = np.append(self._Ur_means[batch],
                                              np.zeros((batch_length, 1)),
                                              axis=1)
            self._Ut_means[batch] = np.append(self._Ut_means[batch],
                                              np.zeros((batch_length, 1)),
                                              axis=1)

        # 2) Loop through the dataframe and sum the reward/replay over each episode
        agent_events['Ur_mean'] = Ur_mean  # [1:].reset_index(drop=True)  # We drop the zeroth row of Ur_means
        self._Ur_means[batch][:, -1] = agent_events.groupby(['ep'])['Ur_mean'].last()  # The final mean max U-value
        agent_events['Ut_mean'] = Ut_mean  # [1:].reset_index(drop=True)  # We drop the zeroth row of Ur_means
        self._Ut_means[batch][:, -1] = agent_events.groupby(['ep'])['Ut_mean'].last()  # The final mean max U-value
        return

    def __compute_reward_rate__(self, batch: str) -> None:
        """
        Computes the reward rates and replay rates (if not with_replay) for the current agent data and stores it
        :param batch: The name of the batch of data we're working on
        :return:
        """
        # If replay is considered we work on the full dataframe, otherwise it's only the real steps
        # We only consider from step 1 as step 0 contains the initial state (and NaNs)
        agent_events = self._agent_events.loc[1:, ['iter', 'step', 'r']]
        agent_events['replay'] = -1 * agent_events['step'].diff()  # The number of replay steps *before* a given s
        agent_events = agent_events[agent_events['step'] == 0]
        agent_events['replay'] = agent_events['replay'].shift(-1)  # Now it's *after* a given s
        agent_events['replay'] = agent_events['replay'].fillna(0)  # Bc of the NaN after the last step
        # It's important to only count real rewards, thus we remove virtual ones
        agent_events.loc[agent_events['step'] > 0, 'r'] = 0
        agent_events = self.__cut_window__(agent_events)

        self.__epoch_based_rate_computer__(self._rew_rate, batch, agent_events)
        self.__epoch_based_rate_computer__(self._replay_rate, batch, agent_events, rate='replay')
        return

    def __compute_cumul_rew__(self, batch: str) -> None:
        """
        Computes the cumulative rewards for the current agent data within the previously specified window and stores it
        :param batch: The name of the batch of data we're working on
        :return:
        """
        try:
            # 0) Extracting the batch information from the batch name
            curr_row = pd.DataFrame([[None] * len(self._cumul_rew.columns)], columns=self._cumul_rew.columns)
            for col_name in self._cumul_rew.columns:
                if col_name == 'cumul_rew':
                    continue
                # Now we want to find the col_name in the batch name. There are 2 possible formats: name_strvalue, or
                # nameNumvalue. Normally this is preceeded by an underscore (_name_val or _nameVal), except at the very
                # beginning of the batch name. val_idx will be the index where the value begins
                # 0.a) let's see if _name_val exists
                val_idx = batch.find(f'_{col_name}_') + len(col_name) + 2  # Idx where the variable value begins
                if val_idx - len(col_name) - 2 == -1:
                    # 0.b) If it doesn't, that means that the value might be numeric, so we should retry with _nameVal
                    res = re.search('_' + col_name + r'\d', batch)
                    if res is not None:
                        val_idx = res.start() + len(col_name) + 1
                    else:
                        # 0.c) If we still haven't found it that means the name_val/nameVal is at the beginning of the
                        # batch name. Let's see if we can find it as name_val
                        val_idx = batch.find(f'{col_name}_')
                        if val_idx == 0:
                            # If it is at the very beginning of the batch name, we're good
                            val_idx = len(col_name) + 1
                        else:
                            # 0.d) If it isn't, then nameVal has to be at the beginning of the file name
                            val_idx = len(col_name)

                # Now we can find the end index of the value using the start index
                val_end = batch[val_idx:].find('_')
                if val_end == -1:
                    curr_row[col_name].iloc[0] = batch[val_idx:]
                else:
                    curr_row[col_name].iloc[0] = batch[val_idx:val_idx + val_end]

            # 1) Preparing the table to use, including the index by which we'll cut it up
            agent_events = self._agent_events.loc[1:, ['iter', 'step', 'r']]
            agent_events.loc[agent_events['step'] > 0, 'r'] = 0
            # If we're epoch based, we need to compute the epoch index, and cut the dataframe based on that
            agent_events = self.__cut_window__(agent_events)

            # 3) Computing and storing the sum reward
            curr_row['cumul_rew'] = agent_events['r'].sum()
            self._cumul_rew = pd.concat([self._cumul_rew, curr_row], axis=0)
            return

        except AttributeError:
            return

    # Internal methods of the plotter
    def __find_max_vals__(self, agent_events: pd.core.frame.DataFrame, col_name: str) -> pd.core.frame.DataFrame:
        """
        Finds the maximal values for a given quality function for each state
        :return:
        """
        vals = pd.DataFrame()
        empty_arr = np.empty(agent_events.shape[0])
        empty_arr[:] = np.nan
        if f'{col_name}' in agent_events.columns:
            vals[f'{col_name}'] = agent_events[f'{col_name}']
        else:
            for s_idx in range(self.state_num()):
                if f'{col_name}_{s_idx}_0' in agent_events.columns:
                    cols = [f'{col_name}_{s_idx}_{a_idx}' for a_idx in self.possible_moves(s_idx)]
                    val_max = agent_events[cols].max(axis=1)
                    # For negative values:
                    # val_min = self._agent_events[cols].min(axis=1)
                    # val_max[abs(val_min) > abs(val_max)] = val_min[abs(val_min) > abs(val_max)]
                    vals[f'{col_name}_{s_idx}'] = val_max
                else:
                    vals[f'{col_name}_{s_idx}'] = pd.DataFrame(empty_arr, columns=[f'{col_name}_{s_idx}'])
        return vals

    def __find_state_coord__(self, row_idx: int, col_name: str) -> Tuple[int, int]:
        """
        Finds the coordinates of a state at a given row in the event table
        :param row_idx: What row we are considering
        :param col_name: s or s_prime
        :return: the x and y coordinates of said state
        """
        s = self._agent_events[col_name].iloc[row_idx]
        [x, y] = np.argwhere(self._maze == s)[0]
        return x, y

    def __mask_walls__(self, values: np.ndarray) -> np.ndarray:
        """
        Masks out all the walls with nans
        """
        values[self._maze < 0] = np.nan
        return values

    def __event_to_img__(self, values: pd.core.frame.DataFrame) -> np.ndarray:
        """
        Takes a row from a pandas dataframe, each column of ot containing a value corresponding a state. This row is
        then converted into a numpy array where these values are projected onto the actual maze.
        :param values:
        :return:
        """
        values = values.to_numpy()
        image = np.empty(self._maze.shape)
        image[:] = np.nan
        image[self._maze >= 0] = values
        return image

    def __wall_to_line__(self, it: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes an iteration and produces 2 arrays, with the x and y coordinates of the walls that need to be plotted over
         the image
        :param it: iteration we are in
        :return:
        """
        wall_states = np.array([])
        for col_name in self._events.columns:
            if col_name[0:5] == 'wall_':
                if self._events[col_name].iloc[it] == 1:
                    col_name = col_name[5:]
                    res = re.search('_', col_name)
                    if len(wall_states) == 0:
                        wall_states = np.array([[int(col_name[0:res.start()]), int(col_name[res.start() + 1:])]])
                    else:
                        wall_states = np.append(wall_states,
                                                np.array(
                                                    [[int(col_name[0:res.start()]), int(col_name[res.start() + 1:])]]),
                                                axis=0)
        wall_x = np.array([])
        wall_y = np.array([])
        for w in wall_states:
            # Decoding the states
            coord0 = np.argwhere(self._maze == w[0])[0]
            coord1 = np.argwhere(self._maze == w[1])[0]

            # The algorithm for finding the x and the y coordinates of the walls is the following.
            # 1) Find the midpoint of the wall
            avg = np.mean(np.array([coord0, coord1]), axis=0)
            # 2) The x vector will be determined by the 2nd cooridnate (vertical if it's x.5, horizontal otherwise)
            x = np.array([[avg[1], avg[1]]]) if avg[1] != coord0[1] else np.array([[avg[1] - 0.5, avg[1] + 0.5]])
            # 2) The y vector will be determined by the 1st cooridnate (horizontal if it's x.5, vertical otherwise)
            y = np.array([[avg[0], avg[0]]]) if avg[0] != coord0[0] else np.array([[avg[0] - 0.5, avg[0] + 0.5]])

            # Add it to the output
            if len(wall_x) == 0:
                wall_x = x
                wall_y = y
            else:
                wall_x = np.append(wall_x, x, axis=0)
                wall_y = np.append(wall_y, y, axis=0)

        return wall_x, wall_y

    def __cut_up_DTmaze__(self, rep: pd.core.frame.DataFrame, **kwargs) -> pd.core.frame.DataFrame:
        """
        This function will take a dataframe containing the replay locations/content over the maze, and will cut it up
        into pre-defined sections and sum the data over these sections
        :param rep: teh replay over the maze
        :param kwargs
            rew_change
        :return: the grouped and summed replay
        """
        poi = {'dec_point': np.empty((0, 2), int),
               'reward': np.empty((0, 2), int),
               # 'start': np.empty((0, 2), int),
               'central_arm': np.empty((0, 2), int),
               'left_side': np.empty((0, 2), int),
               'right_side': np.empty((0, 2), int)}

        # Prepare to detect decision points if the walls were restricted
        restrict = np.copy(self._restrict)
        self.__restrict_walls__()  # I'll do this to exclude "apparent" decision points where most decisions lead to walls
        baseline_act_num = self._restrict.sum(axis=2).astype(object)  # number of forbidden actions for each state
        baseline_act_num[2, 2] = np.min(
            baseline_act_num)  # The left side of the first T can have 1 less action than normal states
        baseline_act_num = np.nanmin(self.act_num() - baseline_act_num)

        # Prepare to detect rewards
        win_begin = self._win_begin  # Win begin can be positive or negative integer
        if win_begin < 0:
            win_begin = self._events['ep'].iloc[-1] + win_begin
        elif win_begin == 0:
            win_begin = 1  # Episodes are actually numbered from 1
        win_end = self._win_end  # Win end can be None (till the end) positive (iter num) or negative (end-iter num)
        if win_end is None:
            win_end = self._events['ep'].iloc[-1]
        elif win_end < 0:
            win_end = self._events['ep'].iloc[-1] + win_end
        begin_idx = np.where(self._events['ep'] == win_begin + 1)[0][0]
        end_idx = np.where(self._events['ep'] == win_end - 1)[0][0]

        # Creating the POI
        for y_dim in range(self._maze.shape[0]):
            for x_dim in range(self._maze.shape[1]):
                if self._maze[y_dim, x_dim] == -1:
                    continue

                # 1) Check if it's the start point
                # if np.all(np.array(np.argwhere(self._maze == self._start_pos)[0]) == np.array([y_dim, x_dim])):
                #     poi['start'] = np.append(poi['start'], np.array([[y_dim, x_dim]]), axis=0)
                #     continue

                # 2) Check if it's rewarded at win_begin and win_end - 1
                rew_idx = 0
                done = False
                while f'rew{rew_idx}_pos_x' in self._events.columns:
                    rew_y = self._events[f'rew{rew_idx}_pos_x'].iloc[begin_idx]
                    rew_x = self._events[f'rew{rew_idx}_pos_y'].iloc[begin_idx]
                    if np.all(np.array([rew_y, rew_x]) == np.array([y_dim, x_dim])):
                        # 1) Check if it's the start point
                        poi['reward'] = np.append(poi['reward'], np.array([[y_dim, x_dim]]), axis=0)
                        done = True
                        break
                    rew_y = self._events[f'rew{rew_idx}_pos_x'].iloc[end_idx]
                    rew_x = self._events[f'rew{rew_idx}_pos_y'].iloc[end_idx]
                    if np.all(np.array([rew_y, rew_x]) == np.array([y_dim, x_dim])):
                        # 1) Check if it's the start point
                        poi['reward'] = np.append(poi['reward'], np.array([[y_dim, x_dim]]), axis=0)
                        done = True
                        break
                    rew_idx += 1
                if done:
                    continue

                # 3) If it's not the start nor the rewarded point, then let's see if it's a decision point
                if self.act_num() - np.sum(self._restrict[y_dim, x_dim, :]) > baseline_act_num:
                    poi['dec_point'] = np.append(poi['dec_point'], np.array([[y_dim, x_dim]]), axis=0)
                    continue

                # 4) If it's none of the above then let's just decide which arm it belongs to
                if (x_dim < 3 and not np.all(np.array([y_dim, x_dim]) == np.array([2, 2]))) or np.all(
                        np.array([y_dim, x_dim]) == np.array([0, 3])):
                    poi['left_side'] = np.append(poi['left_side'], np.array([[y_dim, x_dim]]), axis=0)
                    continue
                elif (x_dim > 4) or np.all(np.array([y_dim, x_dim]) == np.array([5, 4])):
                    poi['right_side'] = np.append(poi['right_side'], np.array([[y_dim, x_dim]]), axis=0)
                    continue
                else:
                    poi['central_arm'] = np.append(poi['central_arm'], np.array([[y_dim, x_dim]]), axis=0)
                    continue

        self._restrict = restrict

        rep_poi = pd.DataFrame(columns=['state', 'poi', 'rep'])
        for element in poi:
            cum_rep = 0
            for s in poi[element]:
                if rep_poi.empty:
                    rep_poi = pd.DataFrame([[self._maze[s[0], s[1]], element, rep.loc[s[0], s[1]]]],
                                           columns=rep_poi.columns)
                else:
                    rep_poi = pd.concat(
                        [pd.DataFrame([[self._maze[s[0], s[1]], element, rep.loc[s[0], s[1]]]],
                                      columns=rep_poi.columns), rep_poi], ignore_index=True)

        return rep_poi

    def __status_to_image__(self, it: int) -> np.ndarray:
        """
        It will produce an array reflecting the status of the maze in iteration it. The array will follow the following
        conventions: wall = 0, path = 1, reward = 2 (irrelevant of value), agent = 3. If the agent is in a rewarded
        state, the state will have a value of 3 (agent)
        :param it: the iteration we are in
        :return:
        """
        # wall = nan, path = 0, reward = 1, agent = 2
        image = np.empty(self._maze.shape)
        image[:] = np.nan
        image[self._maze >= 0] = 0
        reward_num = int((self._events.shape[1] - 3) / 4)
        for rew_idx in range(reward_num):
            if self._events[f'rew{rew_idx}_pos_x'].iloc[it] >= 0:
                image[int(self._events[f'rew{rew_idx}_pos_x'].iloc[it]),
                int(self._events[f'rew{rew_idx}_pos_y'].iloc[it])] = 1
        image[int(self._events['agent_pos_x'].iloc[it]), int(self._events['agent_pos_y'].iloc[it])] = 2
        return image

    def __replay_to_image__(self, curr_image: np.ndarray, row_idx: int) -> np.ndarray:
        """
        Takes the last array representing the replayed states (if no replay had taken lace earlier, we simply use an
        array of zeros) and based on the current row_idx (not iter, not step), we add the last replay to this maze
        :param curr_image: The array depicting the last replay step
        :param row_idx: the row idx in the agent event memory table the replay of which we want to depict
        :return:
        """
        max_val = np.nanmax(curr_image)
        x, y = self.__find_state_coord__(row_idx, 's')
        curr_image[x, y] = max_val + 1
        [x, y] = self.__find_state_coord__(row_idx, 's_prime')
        curr_image[x, y] = max_val + 2
        return curr_image

    def __rate_plotter__(self, data: dict, batches: list, ax) -> None:
        """
        Plots the reward or replay rates on a given axis.
        :param data: self._rew_rate or self._replay_rate
        :param batches: the list of batches to plot
        :param ax: the axis on which we are working
        :return:
        """
        rr = data[batches[0]]
        for batch_idx in range(1, len(batches)):
            # Between the different conditions the length of the time axis might differ given that we might replay
            # differently. Thus, the longest time axis needs to be found and the rest needs to be padded
            rr_curr = data[batches[batch_idx]]
            if rr.shape[0] > rr_curr.shape[0]:
                extension = np.repeat(rr_curr[[-1], :], rr.shape[0] - rr_curr.shape[0], axis=0)
                rr_curr = np.append(rr_curr, extension, axis=0)
            elif rr.shape[0] < rr_curr.shape[0]:
                extension = np.repeat(rr[[-1], :], rr_curr.shape[0] - rr.shape[0], axis=0)
                rr = np.append(rr, extension, axis=0)
            rr = np.append(rr, rr_curr, axis=1)

        # 2) Preparing the indices of the cuts whether or not they are epoch or step based
        win_begin = self._win_begin  # Win begin can be positive or negative integer
        if win_begin < 0:
            win_begin = self._events['ep'].iloc[-1] + win_begin
        elif win_begin == 0:
            win_begin = 1  # Episodes are actually numbered from 1
        win_end = self._win_end  # Win end can be None (till the end) positive (iter num) or negative (end-iter num)
        if win_end is None:
            win_end = self._events['ep'].iloc[-1]
        elif win_end < 0:
            win_end = self._events['ep'].iloc[-1] + win_end
        t = range(win_begin, win_end)
        header = pd.MultiIndex.from_product([batches, [f'{idx}' for idx in range(data[batches[0]].shape[1])]],
                                            names=['batch', 'run'])
        df = pd.DataFrame(rr, index=t, columns=header)
        x_name = 'episodes'
        df[x_name] = t

        # Plotting
        df = pd.melt(df, id_vars=[(x_name, '')])
        palette = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, len(batches))]
        sns.lineplot(df, x=(x_name, ''), y='value', hue='batch', ax=ax, palette=palette)

    # Function to handle the data
    def load_events(self, agent_name: str, env_name: str, batch: str, **kwargs):
        """
        Loads the steps of an agent
        :param agent_name: the name of the agent file [csv]
        :param env_name: the name of the enviornment file [csv]. It is needed as the agent does not know the limits of each episode
        :param batch: what batch the current file belongs to [str]
        :param kwargs:
            path: path to the file. If nothing is specified we'll be looking in the working folder
            replay_chunks: a list of indices between which the replay should be plotted. This only makes sense if we're
                handling an experiment with a single large repay event where we want to see the exact evolution of the
                replay
        :return:
        """
        path = kwargs.get('path', None)
        if path is not None:
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                raise FileNotFoundError(f'No directory named {path}')
        else:
            path = './'

        if os.path.isfile(f'{path}{env_name}'):
            self.load_env(env_name, path=path)
        else:
            raise FileNotFoundError(f'No file named {env_name}')

        if os.path.isfile(f'{path}{agent_name}'):
            self._agent_events = pd.read_csv(f'{path}{agent_name}')
        else:
            raise FileNotFoundError(f'No file named {agent_name}')

        # And then we immediately perform some basic data aggregation
        replay_chunks = kwargs.get('chunked_replay', None)
        if replay_chunks is None:
            self.__count_replay__(batch)
            self.__compute_reward_rate__(batch)
            self.__compute_U_dynamics(batch)
            self.__compute_cumul_rew__(batch)
        else:
            for start_idx in range(len(replay_chunks) - 1):
                self.__count_replay_in_chunks__(replay_chunks[start_idx], replay_chunks[start_idx + 1])
        return

    # Plotters
    def plot_events(self, **kwargs):
        """
        Plots the events of the experiment in an animated fashion. It uses 2 distinct plots: one for the maze, the
        replay and the Q values, the other one for the Ur and the Ut values.
        :kwargs:
            start: the epoch/step number to start from
        :return:
        """
        start = kwargs.get('start', 0)
        if start < 0 or start > max(self._events['ep']):
            raise ValueError('Start has to be between 0 and the number of steps/epochs')
        if start > 0:
            start = np.where(self._events['ep'] == start)[0][0]  # This is the iter where the episode starts
            start = np.where(self._agent_events['iter'] == start)[0][
                0]  # This is the row where the episode starts for the agent

        # 0) Preparing the dataframes -- we need the max Q value for each state, and (as of now) the mean H value
        Q_vals = self.__find_max_vals__(self._agent_events, 'Q')
        Ur_vals = self.__find_max_vals__(self._agent_events, 'Ur')
        Ut_vals = self.__find_max_vals__(self._agent_events, 'Ut')
        C_vals = self.__find_max_vals__(self._agent_events, 'C')

        max_vals = np.array([np.nanmax(Q_vals.iloc[start].to_numpy()),  # Qmax
                             np.nanmax(Ur_vals.iloc[start].to_numpy()),  # Ur max
                             np.nanmax(Ut_vals.iloc[start].to_numpy())])  # Ut max
        # For negative values:
        # max_vals = np.array([np.nanmax(abs(Q_vals.iloc[0].to_numpy())),  # Qmax
        #                      np.nanmax(abs(Ur_vals.iloc[0].to_numpy())),  # Ur max
        #                      np.nanmax(abs(Ut_vals.iloc[0].to_numpy()))])  # Ut max

        # max_vals[max_vals == 0] = 1

        # 1) Preparing the Q plots and the H plots
        plt.ion()
        fig_env, ax_env = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        # fig_env, ax_env = plt.subplots(nrows=1, ncols=3, figsize=(4, 10))
        ax_env[0].set_title("Map")
        ax_env[1].set_title("Replay")
        ax_env[2].set_title("C values")
        curr_maze = self.__status_to_image__(self._agent_events['iter'].iloc[start])
        curr_replay = np.zeros(self._maze.shape)
        curr_replay[self._maze == -1] = np.nan
        curr_C = self.__event_to_img__(C_vals.iloc[start])
        axim_env = np.array([ax_env[0].imshow(curr_maze),
                             ax_env[1].imshow(curr_replay),
                             ax_env[2].imshow(curr_C)])  # , vmin=0, vmax=1)])  #######################################
        axim_env[0].autoscale()  # Since here the extremes already appear
        axim_env[1].autoscale()  # This will have to be done in every step if we want the old replay steps to fade away
        axim_env[2].autoscale()  #######################################
        # And then the walls
        curr_walls_x, curr_walls_y = self.__wall_to_line__(self._agent_events['iter'].iloc[start])
        if len(curr_walls_x) != 0:
            for w_idx in range(curr_walls_x.shape[0]):
                ax_env[0].plot(curr_walls_x[w_idx, :], curr_walls_y[w_idx, :], linewidth=5.0, c='w')

        fig_rla, ax_rla = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        # fig_rla, ax_rla = plt.subplots(nrows=1, ncols=3, figsize=(4, 10))
        ax_rla[0].set_title("Q values")
        ax_rla[1].set_title("Ur values")
        ax_rla[2].set_title("Ut values")
        curr_Q = self.__event_to_img__(Q_vals.iloc[start])
        curr_Ur = self.__event_to_img__(Ur_vals.iloc[start])
        curr_Ut = self.__event_to_img__(Ut_vals.iloc[start])
        axim_rla = np.array([ax_rla[0].imshow(curr_Q, vmin=0, vmax=max_vals[0]),
                             ax_rla[1].imshow(curr_Ur, vmin=0, vmax=max_vals[1]),
                             ax_rla[2].imshow(curr_Ut, vmin=0, vmax=max_vals[2])])
        # For negative values:
        # axim_rla = np.array([ax_rla[0].imshow(curr_Q, vmin=-max_vals[0], vmax=max_vals[0]),
        #                      ax_rla[1].imshow(curr_Ur, vmin=-max_vals[1], vmax=max_vals[1]),
        #                      ax_rla[2].imshow(curr_Ut, vmin=-max_vals[2], vmax=max_vals[2])])

        txt = np.empty((*self._maze.shape, 4), dtype=matplotlib.text.Text)  # txt will appear for the Q and the H values
        for idx_x in range(txt.shape[0]):
            for idx_y in range(txt.shape[1]):
                txt[idx_x, idx_y, 0] = ax_env[2].text(idx_y, idx_x, f"{curr_C[idx_x, idx_y]: .2f}",
                                                      ha="center", va="center", color="w")
                txt[idx_x, idx_y, 1] = ax_rla[0].text(idx_y, idx_x, f"{curr_Q[idx_x, idx_y]: .2f}",
                                                      ha="center", va="center", color="w")
                txt[idx_x, idx_y, 2] = ax_rla[1].text(idx_y, idx_x, f"{curr_Ur[idx_x, idx_y]: .2f}",
                                                      ha="center", va="center", color="w")
                txt[idx_x, idx_y, 3] = ax_rla[2].text(idx_y, idx_x, f"{curr_Ut[idx_x, idx_y]: .1f}",
                                                      ha="center", va="center", color="w")
        plt.show()

        # And the R values #############################################################################################
        # R_vals = self.__find_max_vals__('R')
        # var_vals = self.__find_max_vals__('var')
        # DHt_vals = self.__find_max_vals__('DHt')
        #
        # max_vals_R = np.array([np.nanmax(R_vals.iloc[0].to_numpy()),  # Rmax
        #                        np.nanmax(var_vals.iloc[0].to_numpy()),  # var max
        #                        np.nanmax(DHt_vals.iloc[0].to_numpy())])  # DHt max
        # # For negative values:
        # # max_vals_R = np.array([np.nanmax(abs(R_vals.iloc[0].to_numpy())),  # Rmax
        # #                        np.nanmax(abs(std_vals.iloc[0].to_numpy())),  # std max
        # #                        np.nanmax(abs(DHt_vals.iloc[0].to_numpy()))])  # DHt max
        #
        # # max_vals_R[max_vals_R == 0] = 1
        #
        # fig_rew, ax_rew = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        # ax_rew[0].set_title("R values")
        # ax_rew[1].set_title("var values")
        # ax_rew[2].set_title("DHt values")
        # curr_R = self.__event_to_img__(R_vals.iloc[0])
        # curr_var = self.__event_to_img__(var_vals.iloc[0])
        # curr_DHt = self.__event_to_img__(DHt_vals.iloc[0])
        # axim_rew = np.array([ax_rew[0].imshow(curr_R, vmin=0, vmax=max_vals_R[0]),
        #                      ax_rew[1].imshow(curr_var, vmin=0, vmax=max_vals_R[1]),
        #                      ax_rew[2].imshow(curr_DHt, vmin=0, vmax=max_vals_R[2])])
        # # For negative values:
        # # axim_rew = np.array([ax_rew[0].imshow(curr_R, vmin=-max_vals_R[0], vmax=max_vals_R[0]),
        # #                      ax_rew[1].imshow(curr_std, vmin=-max_vals_R[1], vmax=max_vals_R[1]),
        # #                      ax_rew[2].imshow(curr_DHt, vmin=-max_vals_R[2], vmax=max_vals_R[2])])
        #
        # txt_R = np.empty((*self._maze.shape, 3), dtype=matplotlib.text.Text)
        # for idx_x in range(txt_R.shape[0]):
        #     for idx_y in range(txt_R.shape[1]):
        #         txt_R[idx_x, idx_y, 0] = ax_rew[0].text(idx_y, idx_x, f"{curr_R[idx_x, idx_y]: .2f}",
        #                                                 ha="center", va="center", color="w")
        #         txt_R[idx_x, idx_y, 1] = ax_rew[1].text(idx_y, idx_x, f"{curr_var[idx_x, idx_y]: .2f}",
        #                                                 ha="center", va="center", color="w")
        #         txt_R[idx_x, idx_y, 2] = ax_rew[2].text(idx_y, idx_x, f"{curr_DHt[idx_x, idx_y]: .1f}",
        #                                                 ha="center", va="center", color="w")
        ################################################################################################################

        # plt.pause(.001)

        # 2) Looping through the memories
        for row_idx in range(start + 1, self._agent_events.shape[0]):
            it = int(self._agent_events['iter'].iloc[row_idx])
            step = int(self._agent_events['step'].iloc[row_idx])

            # 2.a) If the agent's memory does not correspond to that of the environment, we quit
            # It is important to note here that during replay there's always a mismatch (hence if self > 0 we ignore)
            # and that if a reward is given, the agent is moved, so there's also a mismatch
            # TODO this does not work if I artificially move the agent around, but that should be fixed
            # if step == 0 and \
            #         self._maze[int(self._events['agent_pos_x'].iloc[it]),
            #         int(self._events['agent_pos_y'].iloc[it])] == 0 \
            #         and self._agent_events['s_prime'].iloc[row_idx] != \
            #         self._maze[int(self._events['agent_pos_x'].iloc[it]), int(self._events['agent_pos_y'].iloc[it])]:
            #     raise ValueError("mismatch between agent and environment memory")

            # 2.b) Else we have to see if we perform replay or not
            if step > 0:
                curr_replay = self.__replay_to_image__(curr_replay, row_idx)
            else:
                curr_replay = np.zeros(self._maze.shape)
                curr_replay[self._maze == -1] = np.nan
            curr_maze = self.__status_to_image__(it)
            curr_Q = self.__event_to_img__(Q_vals.iloc[row_idx])
            curr_Ur = self.__event_to_img__(Ur_vals.iloc[row_idx])
            curr_Ut = self.__event_to_img__(Ut_vals.iloc[row_idx])
            max_vals = np.maximum(max_vals, np.array([np.nanmax(Q_vals.iloc[row_idx].to_numpy()),  # Qmax
                                                      np.nanmax(Ur_vals.iloc[row_idx].to_numpy()),  # Ur max
                                                      np.nanmax(Ut_vals.iloc[row_idx].to_numpy())]))  # Ut max
            # For negative values:
            # max_vals = np.maximum(max_vals, np.array([np.nanmax(abs(Q_vals.iloc[row_idx].to_numpy())),  # Qmax
            #                                           np.nanmax(abs(Ur_vals.iloc[row_idx].to_numpy())),  # Ur max
            #                                           np.nanmax(abs(Ut_vals.iloc[row_idx].to_numpy()))]))  # Ut max

            # max_vals[max_vals == 0] = 1
            curr_C = self.__event_to_img__(C_vals.iloc[row_idx])

            # 2.c) Refresh txt
            for idx_x in range(txt.shape[0]):
                for idx_y in range(txt.shape[1]):
                    txt[idx_x, idx_y, 0].set_text(f"{curr_C[idx_x, idx_y]: .2f}")
                    txt[idx_x, idx_y, 1].set_text(f"{curr_Q[idx_x, idx_y]: .2f}")
                    txt[idx_x, idx_y, 2].set_text(f"{curr_Ur[idx_x, idx_y]: .2f}")
                    txt[idx_x, idx_y, 3].set_text(f"{curr_Ut[idx_x, idx_y]: .1f}")

            # 2.d) Refresh plots
            axim_env[0].set_data(curr_maze)
            axim_env[1].set_data(curr_replay)
            axim_env[2].set_data(curr_C)
            axim_env[1].autoscale()
            axim_env[2].autoscale()  #######################################3

            # And then the walls
            curr_walls_x, curr_walls_y = self.__wall_to_line__(it)
            while len(ax_env[0].lines) > 0:
                ax_env[0].lines[0].remove()
            for w_idx in range(curr_walls_x.shape[0]):
                ax_env[0].plot(curr_walls_x[w_idx, :], curr_walls_y[w_idx, :], linewidth=5.0, c='w')

            axim_rla[0].set_data(curr_Q)
            axim_rla[1].set_data(curr_Ur)
            axim_rla[2].set_data(curr_Ut)
            axim_rla[0].set_clim(vmax=max_vals[0])
            axim_rla[1].set_clim(vmax=max_vals[1])
            axim_rla[2].set_clim(vmax=max_vals[2])
            # For negative values:
            # axim_rla[0].set_clim(vmin=-max_vals[0], vmax=max_vals[0])
            # axim_rla[1].set_clim(vmin=-max_vals[1], vmax=max_vals[1])
            # axim_rla[2].set_clim(vmin=-max_vals[2], vmax=max_vals[2])

            # 2.e) Stop
            fig_env.canvas.flush_events()
            fig_rla.canvas.flush_events()
            plt.show()
            # plt.pause(.001)

            # And for the R values #####################################################################################
            # curr_R = self.__event_to_img__(R_vals.iloc[row_idx])
            # curr_var = self.__event_to_img__(var_vals.iloc[row_idx])
            # curr_DHt = self.__event_to_img__(DHt_vals.iloc[row_idx])
            #
            # max_vals_R = np.maximum(max_vals_R, np.array([np.nanmax(R_vals.iloc[row_idx].to_numpy()),  # Qmax
            #                                               np.nanmax(var_vals.iloc[row_idx].to_numpy()),  # Ur max
            #                                               np.nanmax(DHt_vals.iloc[row_idx].to_numpy())]))  # Ut max
            # # For negative values:
            # # max_vals_R = np.maximum(max_vals_R, np.array([np.nanmax(abs(R_vals.iloc[row_idx].to_numpy())),  # Qmax
            # #                                               np.nanmax(abs(std_vals.iloc[row_idx].to_numpy())),  # Ur max
            # #                                               np.nanmax(abs(DHt_vals.iloc[row_idx].to_numpy()))]))  # Ut max
            #
            # # max_vals[max_vals == 0] = 1
            #
            # for idx_x in range(txt_R.shape[0]):
            #     for idx_y in range(txt_R.shape[1]):
            #         txt_R[idx_x, idx_y, 0].set_text(f"{curr_R[idx_x, idx_y]: .2f}")
            #         txt_R[idx_x, idx_y, 1].set_text(f"{curr_var[idx_x, idx_y]: .2f}")
            #         txt_R[idx_x, idx_y, 2].set_text(f"{curr_DHt[idx_x, idx_y]: .1f}")
            #
            # axim_rew[0].set_data(curr_R)
            # axim_rew[1].set_data(curr_var)
            # axim_rew[2].set_data(curr_DHt)
            # axim_rew[0].set_clim(vmax=max_vals_R[0])
            # axim_rew[1].set_clim(vmax=max_vals_R[1])
            # axim_rew[2].set_clim(vmax=max_vals_R[2])
            # # For negative values:
            # # axim_rew[0].set_clim(vmin=-max_vals_R[0], vmax=max_vals_R[0])
            # # axim_rew[1].set_clim(vmin=-max_vals_R[1], vmax=max_vals_R[1])
            # # axim_rew[2].set_clim(vmin=-max_vals_R[2], vmax=max_vals_R[2])
            # fig_rew.canvas.flush_events()
            ############################################################################################################

    def plot_U_dynamics(self, batches: list, **kwargs) -> None:
        """
        Plots the U-dynamics
        :param batches: a list of the names of the batches we want to compare
        :param kwargs:
            save_img: do I want to save the output fig
            path: if save_img, where do I want to save (default: ./)
            label: if save_img, what tag should I attach to the figure name
            log_U: using a log-scale for the U values
        """
        fig_U, axes_U = plt.subplots(2, 1, figsize=(15, 9))
        axes_U[0].set_title('Reward uncertainty')
        axes_U[0].set_ylabel('Mean Ur-max in the maze')
        axes_U[1].set_title('Transition uncertainty')
        axes_U[1].set_ylabel('Mean Ut-max in the maze')

        axes_U[0].set_xlabel('Episodes')
        axes_U[1].set_xlabel('Episodes')
        log_U = kwargs.get('log_U', True)
        if log_U:
            axes_U[0].set(yscale="log")
            axes_U[1].set(yscale="log")

        self.__rate_plotter__(self._Ur_means, batches, axes_U[0])
        self.__rate_plotter__(self._Ut_means, batches, axes_U[1])

        # Saving the fig if we need to
        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            plt.savefig(f'{path}U_dynamics_{label}.pdf', format="pdf", bbox_inches="tight")

    def plot_reward_rates(self, batches: list, **kwargs) -> None:
        """
        Plots the reward rates over the steps or over the epochs. If replay steps are not considered in the x axis, a
        replay-rate subplot is also produced.
        :param batches: a list of the names of the batches we want to compare
        :param kwargs:
            save_img: do I want to save the output fig
            path: if save_img, where do I want to save (default: ./)
            label: if save_img, what tag should I attach to the figure name
        :return:
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 9))
        axes[1].set_title('Replay rates')
        axes[0].set_title('Reward rates')

        axes[0].set_xlabel('Episodes')
        axes[0].set_ylabel(f'Reward / steps in epoch')
        axes[1].set_xlabel('Episodes')
        axes[1].set_ylabel(f'Replay steps / real steps in epoch')

        # Creating a dataframe for seaborn to work with
        self.__rate_plotter__(self._rew_rate, batches, axes[0])
        self.__rate_plotter__(self._replay_rate, batches, axes[1])
        # plt.show(block=False)

        # Saving the fig if we need to
        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            plt.savefig(f'{path}rew_rate_{label}.pdf', format="pdf", bbox_inches="tight")

    def plot_replay_comp(self, to_plot: str, batches: list, **kwargs) -> None:
        """
        Plots the difference between average replay/stopping frequencies in each state between 2 batches.
        :param to_plot: 'loc' for where the agent stopped to replay, 'content' for what the agent reoplayed
        :param batches: the 2 batches to compare
        :param kwargs:
            save_img: do I want to save the output fig
            path: if save_img, where do I want to save (default: ./)
            label: if save_img, what tag should I attach to the figure name
            bar: if True, we will plot a comparative bar plot instead
            lims: [vmin, vmax] for plotting
        :return:
        """
        if len(batches) != 2:
            raise ValueError("One can only compare 2 batches at a time.")
        if to_plot not in ['loc', 'content']:
            raise ValueError('to_plot has to be either "loc" or "content"')
        if to_plot == 'loc':
            title = 'Stopped to replay at'
        else:
            title = 'Replayed'
        bar = kwargs.get('bar', False)
        if self._win_begin is not None or self._win_end is not None:
            beg = self._win_begin
            if self._win_begin < 0:
                beg = f'(end - {-self._win_begin})'
            elif beg == 0:
                beg = 1  # Episodes are numbered from 1
            end = self._win_end
            if self._win_end is None:
                end = 'the end'
            elif self._win_end < 0:
                end = f'(end - {-self._win_end})'
            title = title + f' between {beg} and {end}'
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))

        df = None
        if to_plot == 'loc':
            if not bar:
                df = pd.DataFrame(self.__mask_walls__(np.mean(self._stopped_at[batches[-1]], axis=2))) - \
                     pd.DataFrame(self.__mask_walls__(np.mean(self._stopped_at[batches[0]], axis=2)))
                lab = 'expected stops'
                if self._norm_rep is not None:
                    lab = 'likelihood of stopping'
            else:
                for rep in range(self._stopped_at[batches[0]].shape[2]):
                    df_1 = self.__cut_up_DTmaze__(pd.DataFrame(self._stopped_at[batches[0]][:, :, rep]))
                    df_1['batch'] = [batches[0]] * len(df_1)
                    df_2 = self.__cut_up_DTmaze__(pd.DataFrame(self._stopped_at[batches[-1]][:, :, rep]))
                    df_2['batch'] = [batches[-1]] * len(df_2)
                    df = pd.concat([df, df_1, df_2],
                                   ignore_index=True)
        elif to_plot == 'content':
            if not bar:
                df = pd.DataFrame(self.__mask_walls__(np.mean(self._replayed[batches[-1]], axis=2))) - \
                     pd.DataFrame(self.__mask_walls__(np.mean(self._replayed[batches[0]], axis=2)))
                lab = 'expected replay'
                if self._norm_rep:
                    lab = 'likelihood of replay'
            else:
                for rep in range(self._replayed[batches[0]].shape[2]):
                    df_1 = self.__cut_up_DTmaze__(pd.DataFrame(self._replayed[batches[0]][:, :, rep]))
                    df_1['batch'] = [batches[0]] * len(df_1)
                    df_2 = self.__cut_up_DTmaze__(pd.DataFrame(self._replayed[batches[-1]][:, :, rep]))
                    df_2['batch'] = [batches[-1]] * len(df_2)
                    df = pd.concat([df, df_1, df_2],
                                   ignore_index=True)

        axes.set_title(f'{title} ({batches[-1]}-{batches[0]})')
        palette = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, len(batches))]
        if not bar:
            lims = kwargs.get('lims', None)
            if lims is not None:
                sns.heatmap(df, ax=axes, center=0, cbar_kws={'label': lab}, cmap='vlag', vmin=lims[0], vmax=lims[1])
            else:
                sns.heatmap(df, ax=axes, center=0, cbar_kws={'label': lab}, cmap='vlag')
        else:
            ax = sns.barplot(df, ax=axes, x='poi', y='rep', hue='batch',
                             order=['left_side', 'central_arm', 'right_side', 'reward', 'dec_point'], palette=palette)
            # add_stat_annotation(ax, data=df, x='poi', y='rep', hue='batch',
            #                     box_pairs=[(('left_side', batches[0]), ('left_side', batches[-1])),
            #                                (('right_side', batches[0]), ('right_side', batches[-1])),
            #                                (('central_arm', batches[0]), ('central_arm', batches[-1])),
            #                                # (('start', batches[0]), ('start', batches[-1])),
            #                                (('reward', batches[0]), ('reward', batches[-1])),
            #                                (('dec_point', batches[0]), ('dec_point', batches[-1])),
            #                                (('left_side', batches[0]), ('right_side', batches[0])),
            #                                (('left_side', batches[-1]), ('right_side', batches[-1])),
            #                                (('left_side', batches[0]), ('central_arm', batches[0])),
            #                                (('left_side', batches[-1]), ('central_arm', batches[-1])),
            #                                (('right_side', batches[0]), ('central_arm', batches[0])),
            #                                (('right_side', batches[-1]), ('central_arm', batches[-1])),
            #                                (('dec_point', batches[0]), ('reward', batches[0])),
            #                                (('dec_point', batches[-1]), ('reward', batches[-1])),
            #                                # (('dec_point', batches[0]), ('start', batches[0])),
            #                                # (('dec_point', batches[-1]), ('start', batches[-1])),
            #                                # (('reward', batches[0]), ('start', batches[0])),
            #                                # (('reward', batches[-1]), ('start', batches[-1]))
            #                                ],
            #                     test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
            pairs = [(('left_side', batches[0]), ('left_side', batches[-1])),
                     (('right_side', batches[0]), ('right_side', batches[-1])),
                     (('central_arm', batches[0]), ('central_arm', batches[-1])),
                     # (('start', batches[0]), ('start', batches[-1])),
                     (('reward', batches[0]), ('reward', batches[-1])),
                     (('dec_point', batches[0]), ('dec_point', batches[-1])),
                     (('left_side', batches[0]), ('right_side', batches[0])),
                     (('left_side', batches[-1]), ('right_side', batches[-1])),
                     (('left_side', batches[0]), ('central_arm', batches[0])),
                     (('left_side', batches[-1]), ('central_arm', batches[-1])),
                     (('right_side', batches[0]), ('central_arm', batches[0])),
                     (('right_side', batches[-1]), ('central_arm', batches[-1])),
                     (('dec_point', batches[0]), ('reward', batches[0])),
                     (('dec_point', batches[-1]), ('reward', batches[-1])),
                     # (('dec_point', batches[0]), ('start', batches[0])),
                     # (('dec_point', batches[-1]), ('start', batches[-1])),
                     # (('reward', batches[0]), ('start', batches[0])),
                     # (('reward', batches[-1]), ('start', batches[-1]))
                     ]
            annotator = Annotator(ax, pairs, data=df, x='poi', y='rep', hue='batch',
                                  order=['left_side', 'central_arm', 'right_side', 'reward', 'dec_point'])
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
            annotator.apply_and_annotate()

        plt.show(block=False)

        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            if not bar:
                plt.savefig(f'{path}comp_rep_{to_plot}_{label}.pdf', format="pdf", bbox_inches="tight")
            else:
                plt.savefig(f'{path}comp_rep_bar_{to_plot}_{label}.pdf', format="pdf", bbox_inches="tight")

    def plot_replay(self, to_plot: str, batches: list, shape: list, **kwargs) -> None:
        """
        Plots the maze with the average replay or stopping frequencies in each state.
        :param to_plot: 'loc' for where the agent stopped to replay, 'content' for what the agent reoplayed
        :param batches: a list of the names of the batches we consider (each in a separate img, shared color scale)
        :param shape: the shape of the subplots
        :param kwargs:
            save_img: do I want to save the output fig
            path: if save_img, where do I want to save (default: ./)
            label: if save_img, what tag should I attach to the figure name
            box_plot: if True, instead of a representation over the whole maze, we will plot
            lims: [vmin, vmax]
        :return:
        """
        if to_plot not in ['loc', 'content']:
            raise ValueError('to_plot has to be either "loc" or "content"')
        if to_plot == 'loc':
            title = 'Stopped to replay at'
        else:
            title = 'Replayed'
        if self._win_begin is not None or self._win_end is not None:
            beg = self._win_begin
            if self._win_begin < 0:
                beg = f'(end - {-self._win_begin})'
            elif beg == 0:
                beg = 1  # Episodes are numbered from 1
            end = self._win_end
            if self._win_end is None:
                end = 'the end'
            elif self._win_end < 0:
                end = f'(end - {-self._win_end})'
            title = title + f' between {beg} and {end}'
        fig, axes = plt.subplots(shape[0], shape[1], figsize=(8 * shape[1], 4 * shape[0]))
        if shape[0] * shape[1] != len(batches):
            raise ValueError('The number of batches has to be the same as the number of subplots')
        elif shape[0] == 1 or shape[1] == 1:
            axes = np.array([axes])  # So that later on I can index it

        # Collecting the final dataframes
        df = [None] * len(batches)  # All te dataframes
        vmin, vmax = None, None
        for batch_idx in range(len(batches)):
            if to_plot == 'loc':
                if not kwargs.get('box_plot', False):
                    df[batch_idx] = pd.DataFrame(self.__mask_walls__(np.mean(self._stopped_at[batches[batch_idx]], axis=2)))
                    lab = 'expected stops'
                    if self._norm_rep is not None:
                        lab = 'likelihood of stopping'
                else:
                    for rep in range(self._stopped_at[batches[batch_idx]].shape[2]):
                        df[batch_idx] = pd.concat([df[batch_idx],
                                                   self.__cut_up_DTmaze__(
                                                       pd.DataFrame(
                                                           self._stopped_at[batches[batch_idx]][:, :, rep]))],
                                                  ignore_index=True)
            elif to_plot == 'content':
                if not kwargs.get('box_plot', False):
                    df[batch_idx] = pd.DataFrame(self.__mask_walls__(np.mean(self._replayed[batches[batch_idx]], axis=2)))
                    lab = 'expected replay'
                    if self._norm_rep:
                        lab = 'likelihood of replay'
                else:
                    for rep in range(self._replayed[batches[batch_idx]].shape[2]):
                        df[batch_idx] = pd.concat([df[batch_idx],
                                                   self.__cut_up_DTmaze__(
                                                       pd.DataFrame(
                                                           self._replayed[batches[batch_idx]][:, :, rep]))],
                                                  ignore_index=True)

            # For normalization purposes
            lims = kwargs.get('lims', None)
            if lims is not None:
                vmin = lims[0]
                vmax = lims[1]
            else:
                if batch_idx == 0:
                    if not kwargs.get('box_plot', False):
                        vmin = np.nanmin(df[batch_idx].to_numpy())
                        vmax = np.nanmax(df[batch_idx].to_numpy())
                    else:
                        vmin = np.nanmin(df[batch_idx].rep)
                        vmax = np.nanmax(df[batch_idx].rep)
                else:
                    if not kwargs.get('box_plot', False):
                        if vmin > np.nanmin(df[batch_idx].to_numpy()):
                            vmin = np.nanmin(df[batch_idx].to_numpy())
                        if vmax < np.nanmax(df[batch_idx].to_numpy()):
                            vmax = np.nanmax(df[batch_idx].to_numpy())
                    else:
                        if vmin > np.nanmin(df[batch_idx].rep):
                            vmin = np.nanmin(df[batch_idx].rep)
                        if vmax < np.nanmax(df[batch_idx].rep):
                            vmax = np.nanmax(df[batch_idx].rep)

        # The actual plotting
        for batch_idx in range(len(batches)):
            idx_x = math.floor(batch_idx / axes.shape[1])
            idx_y = batch_idx % axes.shape[1]
            axes[idx_x, idx_y].set_title(f'{title} ({batches[batch_idx]})')
            if not kwargs.get('box_plot', False):
                sns.heatmap(df[batch_idx], ax=axes[idx_x, idx_y], vmin=vmin, vmax=vmax, cbar_kws={'label': lab}, cmap='viridis')
            else:
                palette = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, df[batch_idx]['poi'].nunique())]
                sns.boxplot(df[batch_idx], ax=axes[idx_x, idx_y], x='poi', y='rep', hue='poi', palette=palette)
                axes[idx_x, idx_y].set(ylim=(vmin, vmax))

        plt.show(block=False)

        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            plt.savefig(f'{path}rep_{to_plot}_{label}.pdf', format="pdf", bbox_inches="tight")

    def plot_rep_vs_visits(self, batches: list, **kwargs) -> None:
        """
        Plots a scatter plot of the number of replay steps in a given state vs the number of visits.
        :param batches: a list of the names of the batches we consider (each in a separate img)
        :param kwargs:
            save_img: do I want to save the output fig
            path: if save_img, where do I want to save (default: ./)
            label: if save_img, what tag should I attach to the figure name
        :return:
        """
        # Assigning types to the states:
        types = np.array([['normal' for _ in range(self._reward.shape[1])] for _ in range(self._reward.shape[0])])
        types[self._reward > 0] = 'reward'

        restriction = self._restrict.sum(axis=2)
        restriction = restriction.astype(float)
        restriction[self._maze < 0] = np.nan
        types[restriction < np.nanmedian(restriction)] = 'branch'
        types = types[self._maze >= 0]

        # Collecting the final dataframes
        data = {'visits': [], 'replay': [], 'batch': [], 'type': []}
        for batch_idx in range(len(batches)):
            batch = batches[batch_idx]
            visits = self._crossed[batch]
            visits = np.sum(visits, axis=2)/visits.shape[2]
            visits = visits[self._maze >= 0]

            rep = self._replayed[batch]
            rep = np.sum(rep, axis=2)/rep.shape[2]
            rep = rep[self._maze >= 0]
            # rep = rep[visits <= 500]            ######## VERY arbitrary

            data['replay'] = data['replay'] + rep.tolist()
            # visits = visits[visits <= 500]     ######## VERY arbitrary
            data['visits'] = data['visits'] + visits.tolist()

            data['batch'] = data['batch'] + [batch]*len(visits)
            data['type'] = data['type'] + types.tolist()

        # The actual plotting
        data = pd.DataFrame(data)
        viridis = plt.get_cmap('viridis')
        hue_levels = data['batch'].unique()
        colors = [viridis(i) for i in np.linspace(0, 1, len(hue_levels))]
        palette = dict(zip(hue_levels, colors))
        # print(palette)
        g = sns.FacetGrid(data[data['type'] == 'normal'], col='batch', col_wrap=2, hue='batch', palette=palette)
        g.map(sns.regplot, 'visits', 'replay', order=1, lowess=False)
        # g.add_legend()
        # Get unique batches and types
        types = data['type'].unique()

        for batch_idx in range(len(batches)):
            ax = g.axes.flat[batch_idx]
            data_subset = data[(data['type'] == 'normal') & (data['batch'] == batches[batch_idx])]
            if len(data_subset) > 1:
                res = linregress(data_subset['visits'], data_subset['replay'])
                slope = res.slope
                pval = res.pvalue

                ax.text(
                    0.95, 0.05,
                    f"slope = {slope:.2f}\np = {pval:.5f}",
                    transform=ax.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.6)
                )

        plt.show(block=False)

        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            if label != '':
                label = '_' + label
            plt.savefig(f'{path}rep_vs_visits{label}.pdf', format="pdf", bbox_inches="tight")

        plt.close()

        g = sns.FacetGrid(data[data['type'] == 'normal'], col='batch', col_wrap=2, hue='batch', palette=palette)
        g.map(sns.scatterplot, 'visits', 'replay')

        # Define marker styles per type
        markers = ['s', 'D']
        types = ['branch', 'reward']
        marker_map = dict(zip(types, markers))
        for batch_idx in range(len(batches)):
            ax = g.axes.flat[batch_idx]
            for type_idx in range(len(types)):
                data_subset = data[(data['type'] == types[type_idx]) & (data['batch'] == batches[batch_idx])]
                col = palette[batches[batch_idx]]
                ax.scatter(data_subset['visits'], data_subset['replay'], color=col, marker=markers[type_idx])

        plt.show(block=False)

        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            if label != '':
                label = '_' + label
            plt.savefig(f'{path}rep_vs_visits_full{label}.pdf', format="pdf", bbox_inches="tight")

        plt.close()

    # Statistics
    def __DT_left_corridor__(self, data: np.ndarray) -> list:
        """
        Returns the elements of the left corridor in the DT-maze
        """
        left_corridor = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0, 0]], dtype='bool')
        return list(data[left_corridor])

    def __DT_right_corridor__(self, data: np.ndarray) -> list:
        """
        Returns the elements of the left corridor in the DT-maze
        """
        right_corridor = np.array([[0, 0, 0, 0, 0, 1, 1, 1],
                               [0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 1, 1, 1, 1]], dtype='bool')
        return list(data[right_corridor])

    def DT_compare_left_right(self, to_comp: str, batches: list, save_path: str, **kwargs):
        """
        The goal of this function is to compare the mean replay between 2 pre-defined regions in the maze
        """
        data = None
        if to_comp not in ['loc', 'content']:
            raise ValueError('to_plot has to be either "loc" or "content"')
        if to_comp == 'loc':
            data_type = 'pauses'
            data = self._stopped_at
        else:
            data_type = 'replay steps'
            data = self._replayed
        if self._win_begin is not None or self._win_end is not None:
            beg = self._win_begin
            if self._win_begin < 0:
                beg = f'(end - {-self._win_begin})'
            elif beg == 0:
                beg = 1  # Episodes are numbered from 1
            end = self._win_end
            if self._win_end is None:
                end = 'the end'
            elif self._win_end < 0:
                end = f'(end - {-self._win_end})'

        # Taking care of the save path
        if save_path[-1] != '/':
            save_path = f'{save_path}/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            print(f'Folder {save_path} created.')
        label = kwargs.get('label', '')
        if label != '':
            label = f'_{label}'

        # Here we will do the statistics individually for each and every batch
        df_diff = pd.DataFrame(columns=['w', 'diff'])
        with open(f'{save_path}statsLeftRight{label}.txt', 'w') as txt_file:
            print(f'Comparing {data_type} between {beg} and {end}:\n', file=txt_file)
            for batch_idx in range(len(batches)):
                print(f'\tBatch {batches[batch_idx]}:\n', file=txt_file)
                df = pd.DataFrame(columns=['left', 'right'])
                data_batch = data[batches[batch_idx]]
                for ep_idx in range(data_batch.shape[2]):
                    left = self.__DT_left_corridor__(data_batch[:, :, ep_idx])
                    right = self.__DT_right_corridor__(data_batch[:, :, ep_idx])
                    df_temp = pd.DataFrame({'left': left, 'right': right})
                    df = (df_temp.copy() if df.empty else pd.concat([df, df_temp], axis=0, ignore_index=True, sort=False))
                    means = df_temp.mean()
                    diff_temp = pd.DataFrame({'w': [float(batches[batch_idx][batches[batch_idx].find('_')+1:])],
                                               'diff': [abs(means['left']-means['right'])]})
                    df_diff = (diff_temp.copy() if df_diff.empty else pd.concat([df_diff, diff_temp], axis=0, ignore_index=True, sort=False))
                stats = scipy.stats.ranksums(df['left'], df['right'])
                print(f'\t\tLeft vs Right (Wilcoxon rank sum): \tp = {stats[1]}\tstat = {stats[0]}\n\n', file=txt_file)

        # Fitting a linear to the differences and plotting it
        plt.figure()
        sns.regplot(data=df_diff, x='w', y='diff', order=1, lowess=False, scatter=False)
        sns.boxplot(data=df_diff, x='w', y='diff', width = 0.5)
        res = linregress(df_diff['w'], df_diff['diff'])
        slope = res.slope
        pval = res.pvalue
        ax = plt.gca()
        plt.text(
            0.95, 0.85,
            f"slope = {slope:.2f}\np = {pval:.5f}",
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.6)
        )

        plt.show(block=False)
        plt.savefig(f'{save_path}diff_means{label}.pdf', format="pdf", bbox_inches="tight")

        plt.figure()
        df_diff_comp = df_diff[df_diff['w'].isin([0, 10])]
        sns.boxplot(data=df_diff_comp, x='w', y='diff', width=0.5)
        ax = plt.gca()
        annotator = Annotator(ax, [(0, 10)], data=df_diff_comp, x='w', y='diff')
        annotator.configure(test='Mann-Whitney', text_format='full', loc='inside')
        annotator.apply_and_annotate()

        plt.show(block=False)
        plt.savefig(f'{save_path}diff_means_zoomed{label}.pdf', format="pdf", bbox_inches="tight")

    def __compute_distances__(self, start: tuple[int, int]) -> np.ndarray:
        rows, cols = self._maze.shape
        distances = np.full((rows, cols), np.inf)  # default: unreachable
        visited = np.zeros((rows, cols), dtype=bool)

        # Only start if the starting point is not a wall
        if self._maze[start] == -1:
            return distances

        # BFS queue: stores (row, col, dist)
        q = deque([(start[0], start[1], 0)])
        distances[start] = 0
        visited[start] = True

        # 4-neighborhood directions (up, down, left, right)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while q:
            r, c, d = q.popleft()

            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if (0 <= nr < rows and 0 <= nc < cols  # inside grid
                        and not visited[nr, nc]  # not visited
                        and self._maze[nr, nc] != -1):  # not a wall

                    visited[nr, nc] = True
                    distances[nr, nc] = d + 1
                    q.append((nr, nc, d + 1))

        # keep walls as -1
        distances[self._maze == -1] = -1
        return distances

    def __DT_distance_from_rew__(self, data: np.ndarray) -> list:
        # Prepare to detect rewards
        win_begin = self._win_begin  # Win begin can be positive or negative integer
        if win_begin < 0:
            win_begin = self._events['ep'].iloc[-1] + win_begin
        elif win_begin == 0:
            win_begin = 1  # Episodes are actually numbered from 1
        win_end = self._win_end  # Win end can be None (till the end) positive (iter num) or negative (end-iter num)
        if win_end is None:
            win_end = self._events['ep'].iloc[-1]
        elif win_end < 0:
            win_end = self._events['ep'].iloc[-1] + win_end
        begin_idx = np.where(self._events['ep'] == win_begin + 1)[0][0]
        end_idx = np.where(self._events['ep'] == win_end - 1)[0][0]

        # Detecting all rewards in the first and last episodes
        rew =[]
        distances = []
        rew_idx = 0
        while f'rew{rew_idx}_pos_x' in self._events.columns:
            [x, y] = [self._events[f'rew{rew_idx}_pos_x'].iloc[begin_idx], self._events[f'rew{rew_idx}_pos_y'].iloc[begin_idx]]
            if not np.isnan(x) and not np.isnan(y) and [x, y] not in rew:
                rew.append([x, y])
                distances.append(self.__compute_distances__((x, y)))
            [x, y] = [self._events[f'rew{rew_idx}_pos_x'].iloc[end_idx],
                      self._events[f'rew{rew_idx}_pos_y'].iloc[end_idx]]
            if not np.isnan(x) and not np.isnan(y) and [x, y] not in rew:
                rew.append([x, y])
                distances.append(self.__compute_distances__((x, y)))
            rew_idx += 1
        distances = np.array(distances)
        distances = np.min(distances, axis=0)

        return [distances[self._maze >= 0], data[self._maze >= 0]]

    def __DT_distance_from_start__(self, data: np.ndarray) -> list:
        # We assume the start point never changes
        [x, y] = [self._events['agent_pos_x'].iloc[0],
                  self._events['agent_pos_y'].iloc[0]]  # Let's assume the agent started from the start position
        distances = self.__compute_distances__((x, y))

        return [distances[self._maze >= 0], data[self._maze >= 0]]

    def __DT_distance_from_branching_point__(self, data: np.ndarray) -> list:
        distances = []
        branching_points = [[0, 4], [2, 3], [5, 3]]  # This is manually defined for the maze
        for [x, y] in branching_points:
            distances.append(self.__compute_distances__((x, y)))
        distances = np.array(distances)
        distances = np.min(distances, axis=0)

        return [distances[self._maze >= 0], data[self._maze >= 0]]

    def DT_plot_biases(self, to_comp: str, batches: list, save_path: str, **kwargs):
        """
        The goal of this function is to compare the mean replay between 2 pre-defined regions in the maze
        """
        data = None
        if to_comp not in ['loc', 'content']:
            raise ValueError('to_plot has to be either "loc" or "content"')
        if to_comp == 'loc':
            data_type = 'pauses'
            data = self._stopped_at
        else:
            data_type = 'replay steps'
            data = self._replayed
        if self._win_begin is not None or self._win_end is not None:
            beg = self._win_begin
            if self._win_begin < 0:
                beg = f'(end - {-self._win_begin})'
            elif beg == 0:
                beg = 1  # Episodes are numbered from 1
            end = self._win_end
            if self._win_end is None:
                end = 'the end'
            elif self._win_end < 0:
                end = f'(end - {-self._win_end})'

        # Taking care of the save path
        if save_path[-1] != '/':
            save_path = f'{save_path}/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            print(f'Folder {save_path} created.')
        label = kwargs.get('label', '')
        if label != '':
            label = f'_{label}'

        # Here we will do the statistics individually for each and every batch
        df = pd.DataFrame(columns=['batch', 'reward', 'start', 'branching point', data_type])
        for batch_idx in range(len(batches)):
            data_batch = data[batches[batch_idx]]
            for ep_idx in range(data_batch.shape[2]):
                rew_dist = self.__DT_distance_from_rew__(data_batch[:, :, ep_idx])
                start_dist = self.__DT_distance_from_start__(data_batch[:, :, ep_idx])
                bp_dist = self.__DT_distance_from_branching_point__(data_batch[:, :, ep_idx])
                if not np.all(rew_dist[1] == start_dist[1]) or not np.all(rew_dist[1] == bp_dist[1]):
                    raise RuntimeError('Mismatch between the replay matrices')
                df_temp = pd.DataFrame({'batch': [batches[batch_idx]]*len(rew_dist[0]), 'reward': rew_dist[0],
                                        'start': start_dist[0],
                                        'branching point': bp_dist[0], data_type: rew_dist[1]})
                df = (df_temp.copy() if df.empty else pd.concat([df, df_temp], axis=0, ignore_index=True, sort=False))

        # Fitting a linear to the differences and plotting it
        order = ['start', 'reward', 'branching point']
        plt.figure()
        df_plot = pd.melt(df, id_vars=['batch', data_type], var_name='from', value_name='distance')
        viridis = plt.get_cmap('viridis')
        hue_levels = df_plot['batch'].unique()
        colors = [viridis(i) for i in np.linspace(0, 1, len(hue_levels))]
        palette = dict(zip(hue_levels, colors))
        g = sns.FacetGrid(df_plot, col='batch', hue='batch', palette=palette, row='from', row_order=order)
        g.map(sns.regplot, 'distance', data_type, order=1, lowess=False, scatter=False)
        g.map(sns.boxplot, 'distance', data_type, width=0.5, order=np.sort(df_plot['distance'].unique()))

        for row_idx in range(len(order)):
            for batch_idx in range(len(batches)):
                ax = g.axes[row_idx, batch_idx]
                # The following part is not adaptive at all, and is highly specific of one experiment ##################
                ax_text = ax.title.get_text()
                distance_from = ax_text[ax_text.find('from = ')+7:ax_text.find(' | ')]
                # w = ax_text[ax_text.find('from = ')+7:ax_text.find(' | ')]
                batch_name = ax_text[ax_text.find('batch = ')+8:]
                data_subset = df_plot[(df_plot['from'] == distance_from) & (df_plot['batch'] == batch_name)]
                if len(data_subset) > 1:
                    res = linregress(data_subset['distance'], data_subset[data_type])
                    slope = res.slope
                    pval = res.pvalue

                    ax.text(
                        0.95, 0.85,
                        f"slope = {slope:.2f}\np = {pval:.2e}",
                        transform=ax.transAxes,
                        ha='right',
                        va='bottom',
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.6)
                    )
        plt.show(block=False)
        plt.savefig(f'{save_path}distances{label}.pdf', format="pdf", bbox_inches="tight")


def plot_cumul_rew_matrix(self, params: list, **kwargs):
        """
        Plots (and potentially stores) a matrix representation of the max (or mean) reward rates over a series of runs
        :param params: The parameters constituting the x and y axes of the produced matrix [list of 2 strings]
        :param kwargs:
            save_img: do I want to save the output fig
            path: if save_img, where do I want to save (default: ./)
            label: if save_img, what tag should I attach to the figure name
            method: if "mean" then mean cumulative reward rates will be computed instead of max (default). In this case
                the matrix annotation will change too: instead of the parameters of the best model, we'll use the avg
                reward rate as label
        :return:
        """
        # 1) Preparing the figure
        fig, ax = plt.subplots(1, 1, figsize=(1.8 * len(self._cumul_rew[params[0]].unique()),
                                              1.5 * len(self._cumul_rew[params[1]].unique())))
        titl = 'episode'
        beg = self._win_begin
        if self._win_begin < 0:
            beg = f'(end - {-self._win_begin})'
        elif beg == 0:
            beg = 1  # Episodes are numbered from 1
        end = self._win_end
        if self._win_end is None:
            end = 'the end'
        elif self._win_end < 0:
            end = f'(end - {-self._win_end})'
            titl = titl + 's'

        # 2) Preparing the data -- we take the maximum for each group
        method = kwargs.get('method', 'max')
        if method == 'max':
            ax.set_title(f'Max cumulative rewards between {titl} {beg} and {end}')
            # 2.a) We can find the max of the cum reward rates
            cumul_mat = self._cumul_rew.loc[
                self._cumul_rew.groupby(params)['cumul_rew'].transform(max) == self._cumul_rew['cumul_rew']]
            # Since this returns *all* maxima, we might want to drop the duplicates, otherwise it'll be impossible to plot
            cumul_mat = cumul_mat.drop_duplicates(params, keep='first')
            # Now we need to do pandas magic to make it understand non-numeric axis values
            cumul_mat_heatmap = cumul_mat.pivot(columns=params[0], index=params[1], values='cumul_rew')
            # And we need to make an identically shaped annotation matrix
            cumul_mat_annot = None
            for col_name in cumul_mat.columns:
                if col_name in params or col_name == 'cumul_rew':
                    continue
                if cumul_mat_annot is None:
                    cumul_mat_annot = f'{col_name}=' + cumul_mat.pivot(columns=params[0], index=params[1],
                                                                       values=col_name)
                else:
                    cumul_mat_annot += f'\n {col_name}=' + cumul_mat.pivot(columns=params[0], index=params[1],
                                                                           values=col_name)
            lab = 'max cumulative reward'
            annot_format = ''
        elif method == 'mean':
            ax.set_title(f'Mean cumulative rewards between {titl} {beg} and {end}')
            # 2.b) Or we can just plot the means
            cumul_mat = self._cumul_rew.groupby(params)['cumul_rew'].mean()
            # Then we basically need to "un-groupby" it
            cumul_mat_heatmap = cumul_mat.reset_index(params)
            # And then make sure that pandas understands non-numeric axis values
            cumul_mat_heatmap = cumul_mat_heatmap.pivot(columns=params[0], index=params[1], values='cumul_rew')
            # Finally the annotation matrix, here it will only contain the reward values
            cumul_mat_annot = cumul_mat_heatmap
            lab = 'mean cumulative reward'
            annot_format = '.2f'
        else:
            raise ValueError('Method has to be "max" or "mean".')
        sns.heatmap(cumul_mat_heatmap, annot=cumul_mat_annot, cbar_kws={'label': lab}, fmt=annot_format, cmap='viridis')
        ax.invert_yaxis()
        plt.show(block=False)

        # 3) Saving if necessary
        if kwargs.get('save_img', False):
            path = kwargs.get('path', './')
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
            label = kwargs.get('label', '')
            plt.savefig(f'{path}{method}_cumul_rew_{params[0]}_{params[1]}_{label}.pdf', format="pdf",
                        bbox_inches="tight")
