import matplotlib
import numpy as np
from typing import Tuple
import random
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import math
import seaborn as sns
import re


class Env:
    """
    The environment class will contain all information regarding the environment, as well as it will be responsible for
    the generation of observations for the agent by reacting to the actions of the former.

    The environment is created independently of the agent, thus it can be easily replaced by a different class, or
    further subclasses may be introduced with ease, as long as the communication between it and the agent remains
    undisturbed.
    """

    def __init__(self, **kwargs):
        """
        General constructor of the environment, simply defining the most basic elements of it
        """
        # The encoding of the entire action space (all possible actions, m dimensions)
        self._act = np.array([])
        # The encoding of the maze -- 0 represents a wall, a number represents an available state
        self._maze = np.array([])
        # What is the value of each state defined above
        self._reward = np.array([])
        # What is the likelihood of getting a reward in each state
        self._reward_prob = np.array([])
        # Where te agent is right now
        self._agent_pos = np.array([])
        # Are there any forbidden actions (following the coding of _act) -- for every state specify a vector of
        # length m. 0 means no restriction, 1 means forbidden action in said state
        self._restrict = np.array([])
        # Do I restrict bumping into walls?
        self._forbidden_walls = False
        # The walls that we might slip in between states. Operates the same as restrict
        self._walls = np.array([])
        # Probability of slipping (stochastic transitioning)
        self._slip_prob = 0
        # About storing the data
        self._save_env = False
        self._events = None
        self._start_pos = None
        self._episode_idx = 0  # For saving purposes
        return

    # Hidden functions for several upkeep purposes

    def __restrict_walls__(self) -> None:
        """
        Restricts bumping into a wall, let that be explicit 0 or just out of bounds
        """
        for x in range(self._maze.shape[0]):
            for y in range(self._maze.shape[1]):
                if self._maze[x, y] >= 0:
                    for a in range(len(self._act)):
                        [x_prime, y_prime] = self.__next_state__(x, y, a)
                        if self.__check_out_of_bounds__(x_prime, y_prime) or self._maze[x_prime, y_prime] == -1:
                            self._restrict[x, y, a] = 1

    def __check_out_of_bounds__(self, x: int, y: int) -> bool:
        """
        See if an (x, y) coordinate pair is out of bounds on the map, or is if a forbidden filed (i.e. wall)
        :param x: x coordinate
        :param y: y coordinate
        :return: we are out of bounds (True) or not (False)
        """
        if x < 0 or x >= self._maze.shape[0] or y < 0 \
                or y >= self._maze.shape[1] or self._maze[x, y] == -1:
            return True
        return False

    def __next_state__(self, x: int, y: int, a: int) -> np.ndarray:
        """
        Tells us the label and the coordinates of the next state if we take action a in state s (stays in s if the
        action is impossible)
        :param x: the x coordinate we're in
        :param y: the y coordinate we're in
        :param a: the action chosen
        :return: the coordinates of the arrival state
        """
        x_prime, y_prime = x, y
        if np.sum(self._walls) == 0 or self._walls[x, y, a] == 0:
            [x_prime, y_prime] = np.array([x, y]) + self._act[a]
        return np.array([x_prime, y_prime]).astype(int)

    def __slip__(self, x: int, y: int, x_prime: int, y_prime: int) -> np.array:
        """
        In case of a slippery maze, this function will implement how slip should happen. The basic idea is that we take
        a non-forbidden step from s_prime that doesn't lead us back to s. Watch out, this can happen recursively!
        :param x, y: starting state
        :param x_prime, y_prime: arrival state before slipping
        :return: arrival state after slipping in coordinates
        """
        # First let's decide if we slip or not
        if np.random.uniform(0, 1) >= self._slip_prob:
            return np.array([x_prime, y_prime]).astype(int)

        a_poss = self.possible_moves(self._maze[x_prime, y_prime])
        a_poss_filt = np.copy(a_poss)  # This is the actual a_poss without the action(s) that'd take us back

        # Getting rid of a move that would possibly take us back
        for a in a_poss:
            if np.all(np.array([x, y]) == self.__next_state__(x_prime, y_prime, a)):
                np.delete(a_poss_filt, a_poss_filt == a)

        # Taking the random step
        a = np.random.choice(a_poss_filt)
        [x_fin, y_fin] = self.__next_state__(x_prime, y_prime, a)

        # If we were to go out of bounds, or bump into a wall, stay in place instead:
        if self.__check_out_of_bounds__(x_fin, y_fin):
            x_fin, y_fin = x_prime, y_prime

        # And then slip on recursively
        [x_fin, y_fin] = self.__slip__(x_prime, y_prime, x_fin, y_fin)

        return np.array([x_fin, y_fin]).astype(int)

    def __save_step__(self) -> None:
        """
        Saves the current state of the maze by adding a row to the _events memory.
        :return:
        """
        if not self._save_env:
            return

        # 1) Which step are we at
        step = 0
        if len(self._events['iter']) > 0:
            step = self._events['iter'][-1] + 1

        # 2) Format the event to store (we might have more reward columns than needed)
        # iter, agent_pos_x, agent_pos_y, rew0_pos_x, rew0_pos_y, rew0_val, rew0_proba, ...
        self._events['iter'].append(step)
        self._events['ep'].append(self._episode_idx)
        agent = np.argwhere(self._agent_pos == 1)
        self._events['agent_pos_x'].append(agent[0, 0])
        self._events['agent_pos_y'].append(agent[0, 1])
        rewards = np.argwhere(self._reward > 0)
        rew_added = []
        for rew_idx in range(rewards.shape[0]):
            self._events[f'rew{rew_idx}_pos_x'].append(rewards[rew_idx, 0])
            self._events[f'rew{rew_idx}_pos_y'].append(rewards[rew_idx, 1])
            self._events[f'rew{rew_idx}_val'].append(self._reward[rewards[rew_idx, 0], rewards[rew_idx, 1]])
            self._events[f'rew{rew_idx}_var'].append(self._reward_prob[rewards[rew_idx, 0], rewards[rew_idx, 1]])
            rew_added += [f'rew{rew_idx}_pos_x', f'rew{rew_idx}_pos_y', f'rew{rew_idx}_val', f'rew{rew_idx}_var']
        for col_name in self._events:
            if col_name[0:3] == 'rew' and col_name not in rew_added:
                self._events[col_name].append(np.nan)

        # As for the walls
        wall_states = self._maze[np.sum(self._walls, axis=2) > 0]
        walls_added = []
        for s0 in wall_states:
            [x0, y0] = np.argwhere(self._maze == s0)[0]
            for a0 in self._act[self._walls[x0, y0, :] > 0]:
                coord1 = np.array([x0, y0]) + a0
                s1 = self._maze[coord1[0], coord1[1]]
                if f'wall_{s0}_{s1}' in self._events:  # If we don't have it, then we have wall_s1_s0
                    self._events[f'wall_{s0}_{s1}'].append(1)
                walls_added.append(f'wall_{s0}_{s1}')
        for col_name in self._events:
            if col_name[0:5] == 'wall_' and col_name not in walls_added:
                self._events[col_name].append(0)
        return

    def __overwrite_step__(self, x: int, y: int) -> None:
        """
        Overwrites the last stored memory in case the agent was moved (teleported) without a step having taken place.
        :param x: New x coordinate of the agent
        :param y: New y coordinate of the agent
        :return:
        """
        if self._save_env:
            self._events['agent_pos_x'][-1] = x
            self._events['agent_pos_y'][-1] = y
        return

    def print_map(self, filename: str):
        """
        Just a function I use for debugging purposes. It prints the map and saves it as a pdf
        :return:
        """
        image = np.zeros(self._maze.shape)
        image[self._maze >= 0] = 10
        image[self._maze == self._start_pos] = 5.1
        image[self._reward > 0] = 7
        image[self._reward > 1] = 6.5

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        axim = ax.imshow(image, cmap='gnuplot2', vmin=0, vmax=10)

        # Minor ticks
        ax.set_xticks(np.arange(-.5, self._maze.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, self._maze.shape[0], 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

        plt.show(block=False)
        plt.savefig(f'./img/{filename}.pdf', format='pdf', bbox_inches='tight')

    # Getters that will communicate towards the agent

    def state_num(self) -> int:
        """
        Returns the number of total states possible in the environment, where each state means a separate location in
        the maze

        :return: number of possible states
        """
        return np.max(self._maze) + 1

    def act_num(self) -> int:
        """
        Returns the maximum number of possible actions within the maze for any state.

        :return: max number of possible actions
        """
        return len(self._act)

    # Communication towards the agent

    def curr_state(self) -> int:
        """
        Returns the current state (as per understood by the agent) of the agent

        :return: current state of the agent
        """
        return self._maze[self._agent_pos.astype(bool)][0]

    def possible_moves(self, s: int) -> np.ndarray:
        """
        Given a state of query, it computes all the possible available actions, taking into consideration whether
        movement is restricted or not, and whether the agent can try and bump into walls or not

        :param s: current state label, as understood by the agent
        :return: a numpy array of possible actions to choose from (labels follow those of self._act)
        """

        [x, y] = np.argwhere(self._maze == s)[0]
        moves = np.array(range(len(self._act)))
        moves = moves[~self._restrict[x, y, :].astype(bool)]
        # No need to check for self._walls, because if self._forbidden_walls, the code does not allow me to place a wall
        # If on the other hand not self._forbidden_walls, then it does not matter anyway, cuz I can bump into it w/o a
        # restriction
        return moves.astype(int)

    # And receiving communication from the agent

    def reset_agent(self, **kwargs) -> None:
        """
        Simply resets the agent to its starting state
        :param kwargs:
            end_of_episode: is the agent reset because this is the end of an episode?
            back_to_start: should the agent go back to the start
        :return:
        """
        if kwargs.get('back_to_start', True):
            self.place_agent(self._start_pos)
        if kwargs.get('end_of_episode', True):
            self._episode_idx += 1

    def step(self, s: int, a: int) -> Tuple[int, float]:
        """
        Performs a step from state s (as per designated by the agent), taking action a (as per chosen in advance), and
        returns the observed outcome.
        If the action would drive the agent out of bounds or into the wall, the agent stays in place
        If the environment is slippery, the agent might slip (potentially recursively)
        Every time we take a step, the environment's memory is updated.

        :param s: state label, as per understood by the agent
        :param a: action label, following the indexing of self._act
        :return: new state label as per understood by the agent (int), and corresponding reward (float)
        """
        # Let's remove the agent from the starting state
        [x, y] = np.argwhere(self._maze == s)[0]
        self._agent_pos[x, y] = 0

        # Then see where we land
        [x_prime, y_prime] = self.__next_state__(x, y, a)

        # If we were to go out of bounds, or bump into a wall, stay in place instead:
        if self.__check_out_of_bounds__(x_prime, y_prime):
            x_prime, y_prime = x, y

        # Then we might slip (the function will have no effect if slip_prob == 0)
        [x_prime, y_prime] = self.__slip__(x, y, x_prime, y_prime)

        # Arriving at our final destination in the environment
        s_prime = self._maze[x_prime, y_prime]
        self._agent_pos[x_prime, y_prime] = 1

        # Generating reward
        rew = 0
        if not self._restrict[x_prime, y_prime, 0] == 1:
            # If non-restricted, and stay is a valid (and rewarded) action
            if s_prime == s and random.uniform(0, 1) < self._reward_prob[x_prime, y_prime]:  # for any act that stays
                rew = self._reward[x_prime, y_prime]
        else:
            # If restricted, deliver it upon stepping onto the rewarded state
            if random.uniform(0, 1) < self._reward_prob[
                x_prime, y_prime]:  # for any act that brings us to the rew state
                rew = self._reward[x_prime, y_prime]

        # Saving
        self.__save_step__()

        return s_prime, rew

    def place_reward(self, reward_state: int, reward_val: float, reward_prob: float) -> None:
        """
        Places the reward.
        :param reward_state: Where this reward should be placed (state-space representation)
        :param reward_val: Value of this reward
        :param reward_prob: Probability of said reward
        :return: -
        """
        [x, y] = np.argwhere(self._maze == reward_state)[0]
        # Where is the reward and how big is it?
        self._reward[x, y] = reward_val
        # What is the likelihood of getting a reward
        self._reward_prob[x, y] = reward_prob

        # Call the toggle_save, because in case we are saving, adding a new reward means we need to extend the storage
        self.toggle_save(save_on=self._save_env)
        return

    def reset_reward(self) -> None:
        """
        Resets the reward to zero.

        :return: -
        """
        self._reward = np.zeros(self._maze.shape)
        self._reward_prob = np.zeros(self._maze.shape)
        return

    def place_agent(self, init_state: int) -> None:
        """
        A function to place the agent onto state init_state. If saving is on this function will overwrite the location
        of the agent in the last row of the memory.
        :param init_state: the state (understood by the agent) where we should be placed
        """
        [x, y] = np.argwhere(self._maze == init_state)[0]

        # Remove the agent from its old position
        self._agent_pos = np.zeros(self._maze.shape)
        # And then place it onto the new
        self._agent_pos[x, y] = 1

        # Take care of saving by overwriting the last element
        self.__overwrite_step__(x, y)
        return

    def place_wall(self, s0: int, s1: int) -> None:
        """
        A function to place a piece of wall between two neihboring states.
        :param s0: The state on one side of the wall (as understood by the agent)
        :param s1: The state on the other side of the wall (as understood by the agent)
        :return:
        """
        if self._forbidden_walls:
            raise ValueError('If bumping into walls is forbidden, '
                             'the agent will never learn to avoid a freshly added wall!')

        # Decoding the states
        coord0 = np.argwhere(self._maze == s0)[0]
        coord1 = np.argwhere(self._maze == s1)[0]

        # Finding the actions in between
        a0 = np.where(np.all(self._act == coord1 - coord0, axis=1))[0]
        a1 = np.where(np.all(self._act == coord0 - coord1, axis=1))[0]
        if a0 is None or len(a0) == 0 or a1 is None or len(a1) == 0:
            raise ValueError(f'State {s0} and {s1} are not neigboring states, '
                             f'thus we cannot implement a wall between them')

        # Storing the wall
        self._walls[coord0[0], coord0[1], a0] = 1
        self._walls[coord1[0], coord1[1], a1] = 1

        self.toggle_save(save_on=self._save_env)
        return

    def reset_wall(self) -> None:
        """
        A function to delete all walls placed down between states
        :return:
        """
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        return

    # About saving
    def toggle_save(self, **kwargs) -> None:
        """
        Toggles save. If the environment was saving its status so far, it sops doing so. Otherwise, it begins to do so,
        by already storing a snapshot of the current state as well.
        If a new reward has been added recently, we'll increase the size of the memory to accomodate it.
        :param kwargs:
            save_on: If instead of toggling, we want to make sure to turn it on [True] or off [False], we can
        :return:
        """
        save_on = kwargs.get('save_on', not self._save_env)
        if save_on:
            try:
                if f'rew{np.sum(self._reward > 0) - 1}_pos_x' not in self._events:  # We added a new reward
                    for rew_idx in range(np.sum(self._reward > 0)):
                        if f'rew{rew_idx}_pos_x' not in self._events.keys():
                            self._events[f'rew{rew_idx}_pos_x'] = [np.nan] * (len(self._events['agent_pos_x']))
                            self._events[f'rew{rew_idx}_pos_y'] = [np.nan] * (len(self._events['agent_pos_x']))
                            self._events[f'rew{rew_idx}_val'] = [np.nan] * (len(self._events['agent_pos_x']))
                            self._events[f'rew{rew_idx}_var'] = [np.nan] * (len(self._events['agent_pos_x']))
                wall_states = self._maze[np.sum(self._walls, axis=2) > 0]
                for s0 in wall_states:
                    [x0, y0] = np.argwhere(self._maze == s0)[0]
                    for a0 in self._act[self._walls[x0, y0, :] > 0]:
                        coord1 = np.array([x0, y0]) + a0
                        s1 = self._maze[coord1[0], coord1[1]]
                        if f'wall_{s0}_{s1}' not in self._events and f'wall_{s1}_{s0}' not in self._events:
                            self._events[f'wall_{s0}_{s1}'] = [0] * (len(self._events['agent_pos_x']))

            except TypeError:  # There is no such thing as _events yet
                self._events = {'iter': [], 'ep': [], 'agent_pos_x': [], 'agent_pos_y': []}
                for variable_num in range(np.sum(self._reward > 0)):
                    for variable_name in ['pos_x', 'pos_y', 'val', 'var']:
                        self._events[f'rew{variable_num}_{variable_name}'] = []
                wall_states = self._maze[np.sum(self._walls, axis=2) > 0]
                for s0 in wall_states:
                    [x0, y0] = np.argwhere(self._maze == s0)[0]
                    for a0 in self._act[self._walls[x0, y0, :] > 0]:
                        coord1 = np.array([x0, y0]) + a0
                        s1 = self._maze[coord1[0], coord1[1]]
                        if f'wall_{s1}_{s0}' not in self._events:
                            self._events[f'wall_{s0}_{s1}'] = []
            if not self._save_env:
                self._save_env = True
                self.__save_step__()
        else:
            self._save_env = False

    def dump_env(self, **kwargs) -> None:
        """
        Saves everything that we have stored into 2 different files: one for the environment, and one for the events.
        :param kwargs:
            path: [str] the path to save the document. If no path is defined then the current working folder will be
                used
            label: [str] an additional label to add at the end of the output file name.
        :return:
        """
        path = kwargs.get('path', None)
        if path is not None:
            if path[-1] != '/':
                path = f'{path}/'
            if not os.path.isdir(path):
                os.mkdir(path)
        else:
            path = './'
        label = kwargs.get('label', None)
        if label is not None:
            label = f'_{label}'
        else:
            label = ''

        # 1) Save the whole environment
        self._events = pd.DataFrame(self._events)
        self._events = self._events.astype({'iter': 'int', 'ep': 'int'})
        file = open(f'{path}environment{label}.txt', 'wb')
        pickle.dump(self.__dict__, file, 2)
        file.close()

        # 2) Save the events
        # try:
        #     self._events.to_csv(f'{path}environment{label}.csv', sep=',', index=False, encoding='utf-8')
        # except AttributeError:
        #     print('Note: This environment does not store the transpired events, no .csv generated.')

    def load_env(self, file_name: str, **kwargs):
        """
        Loads a previously saved environment
        :param file_name: the name of the environment file [txt]
        :param kwargs:
            path: path to the file. If nothing is specified we'll be looking in the working folder
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

        if os.path.isfile(f'{path}{file_name}'):
            file = open(f'{path}{file_name}', 'rb')
            tmp_dict = pickle.load(file)
            file.close()
            self.__dict__.update(tmp_dict)
        else:
            raise FileNotFoundError(f'No file named {file_name}')


class DTMaze(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays

    DT Maze stands for double T maze
    """

    def __init__(self, **kwargs):
        """
        Constructor of the double T maze class

        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        restricted_dT   -- creates a restricted double-T maze where we can only walk in one direction
                        slip_prob       -- the probability of slipping after a step
                        start_pos       -- the initial position of the agent
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        restricted_dt = kwargs.get('restricted_dT', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: stay, 1: up, 2: right, 3: down, 4: left}
        self._act = np.array(
            [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        self._maze = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                               [8, -1, -1, -1, 9, -1, -1, 10],
                               [11, -1, 12, 13, 14, -1, -1, 15],
                               [16, -1, -1, 17, -1, -1, -1, 18],
                               [19, -1, -1, 20, -1, -1, -1, 21],
                               [22, 23, 24, 25, 26, 27, 28, 29]])
        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        if restricted_dt:
            # In this case I want to simply test what happens if I restrict going backwards
            self._restrict = np.array(
                [[[1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]],
                 [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]],
                 [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]],
                 [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]],
                 [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]],
                 [[1, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0],
                  [1, 0, 1, 0, 0], [1, 1, 0, 0, 0]]])
        else:
            self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        self._forbidden_walls = forbidden_walls
        if forbidden_walls:
            self.__restrict_walls__()
        # Walls to insert
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))

        # Further options
        self._start_pos = kwargs.get('start_pos', None)
        if self._start_pos is not None:
            self.place_agent(self._start_pos)
        return


class DTMazeWide(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays

    DT Maze stands for double T maze
    """

    def __init__(self, **kwargs):
        """
        Constructor of the double T maze class

        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        restricted_dT   -- creates a restricted double-T maze where we can only walk in one direction
                        slip_prob       -- the probability of slipping after a step
                        start_pos       -- the initial position of the agent
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        restricted_dt = kwargs.get('restricted_dT', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: stay, 1: up, 2: right, 3: down, 4: left}
        self._act = np.array(
            [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        # self._maze = np.array([ [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23],
        #                          [ 24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47],
        #                          [ 48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71],
        #                          [ 72,  73,  74, -1, -1, -1, -1, -1, -1, -1, -1, -1,  75,  76,  77, -1, -1, -1, -1, -1, -1,  78,  79,  80],
        #                          [ 81,  82,  83, -1, -1, -1, -1, -1, -1, -1, -1, -1,  84,  85,  86, -1, -1, -1, -1, -1, -1,  87,  88,  89],
        #                          [ 90,  91,  92, -1, -1, -1, -1, -1, -1, -1, -1, -1,  93,  94,  95, -1, -1, -1, -1, -1, -1,  96,  97,  98],
        #                          [ 99, 100, 101, -1, -1, -1, 102, 103, 104, 105, 106, 107, 108, 109, 110, -1, -1, -1, -1, -1, -1, 111, 112, 113],
        #                          [114, 115, 116, -1, -1, -1, 117, 118, 119, 120, 121, 122, 123, 124, 125, -1, -1, -1, -1, -1, -1, 126, 127, 128],
        #                          [129, 130, 131, -1, -1, -1, 132, 133, 134, 135, 136, 137, 138, 139, 140, -1, -1, -1, -1, -1, -1, 141, 142, 143],
        #                          [144, 145, 146, -1, -1, -1, -1, -1, -1, 147, 148, 149, -1, -1, -1, -1, -1, -1, -1, -1, -1, 150, 151, 152],
        #                          [153, 154, 155, -1, -1, -1, -1, -1, -1, 156, 157, 158, -1, -1, -1, -1, -1, -1, -1, -1, -1, 159, 160, 161],
        #                          [162, 163, 164, -1, -1, -1, -1, -1, -1, 165, 166, 167, -1, -1, -1, -1, -1, -1, -1, -1, -1, 168, 169, 170],
        #                          [171, 172, 173, -1, -1, -1, -1, -1, -1, 174, 175, 176, -1, -1, -1, -1, -1, -1, -1, -1, -1, 177, 178, 179],
        #                          [180, 181, 182, -1, -1, -1, -1, -1, -1, 183, 184, 185, -1, -1, -1, -1, -1, -1, -1, -1, -1, 186, 187, 188],
        #                          [189, 190, 191, -1, -1, -1, -1, -1, -1, 192, 193, 194, -1, -1, -1, -1, -1, -1, -1, -1, -1, 195, 196, 197],
        #                          [198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221],
        #                          [222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245],
        #                          [246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]
        #                         ])

        self._maze = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                               [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                               [32, 33, -1, -1, -1, -1, -1, -1, 34, 35, -1, -1, -1, -1, 36, 37],
                               [38, 39, -1, -1, -1, -1, -1, -1, 40, 41, -1, -1, -1, -1, 42, 43],
                               [44, 45, -1, -1, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, 52, 53],
                               [54, 55, -1, -1, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, 62, 63],
                               [64, 65, -1, -1, -1, -1, 66, 67, -1, -1, -1, -1, -1, -1, 68, 69],
                               [70, 71, -1, -1, -1, -1, 72, 73, -1, -1, -1, -1, -1, -1, 74, 75],
                               [76, 77, -1, -1, -1, -1, 78, 79, -1, -1, -1, -1, -1, -1, 80, 81],
                               [82, 83, -1, -1, -1, -1, 84, 85, -1, -1, -1, -1, -1, -1, 86, 87],
                               [88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103],
                               [104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]])

        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        self._forbidden_walls = forbidden_walls
        if forbidden_walls:
            self.__restrict_walls__()
        # Walls to insert
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))

        # Further options
        self._start_pos = kwargs.get('start_pos', None)
        if self._start_pos is not None:
            self.place_agent(self._start_pos)
        return


class DTMazeTight(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays

    DT Maze stands for double T maze
    """

    def __init__(self, **kwargs):
        """
        Constructor of the double T maze class

        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        restricted_dT   -- creates a restricted double-T maze where we can only walk in one direction
                        slip_prob       -- the probability of slipping after a step
                        start_pos       -- the initial position of the agent
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        restricted_dt = kwargs.get('restricted_dT', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: stay, 1: up, 2: right, 3: down, 4: left}
        self._act = np.array(
            [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        self._maze = np.array([[0, 1, 2, 3, 4],
                               [5, -1, 6, -1, 7],
                               [8, 9, 10, 11, 12]])
        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        if restricted_dt:
            self._restrict = np.array(
                [[[1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]],
                 [[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0]],
                 [[1, 1, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0]]])
        else:
            self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        self._forbidden_walls = forbidden_walls
        if forbidden_walls:
            self.__restrict_walls__()
        # Walls to insert
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))

        # Further options
        self._start_pos = kwargs.get('start_pos', None)
        if self._start_pos is not None:
            self.place_agent(self._start_pos)
        return


class Mmaze(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays

    Mmaze stands for mitochondria-like maze. In this maze, I recommend the following setup:
        - place large distal reward on state 28
        - place small proximal reward on state 101
        - place agent on state 57
    This way the distal reward takes EXACTLY 4 times as many steps (32) to reach as the proximal reward (8). After the
    reward change:
        - place a single reward on state 38
    This reward will be 32 steps away from the agent, just like the OG distal reward. Every reward can be approached via
    2 different routes (6 routes altogehter) and for every pair of rewards, there is one route for each that share its
    initial 4 steps.
    """

    def __init__(self, **kwargs):
        """
        Constructor of the double T maze class

        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        slip_prob       -- the probability of slipping after a step
                        start_pos       -- the initial position of the agent
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: stay, 1: up, 2: right, 3: down, 4: left}
        self._act = np.array(
            [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        self._maze = np.array([[0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, 9, 10, -1, 11, 12, 13, -1, 14, 15, 16],
                               [17, -1, 18, -1, 19, -1, 20, -1, 21, -1, 22, -1, 23, -1, 24, -1, 25, -1, 26, -1, 27],
                               [28, -1, 29, -1, 30, -1, 31, -1, 32, -1, 33, -1, 34, -1, 35, -1, 36, -1, 37, -1, 38],
                               [39, -1, 40, -1, 41, -1, 42, -1, 43, -1, 44, -1, 45, -1, 46, -1, 47, -1, 48, -1, 49],
                               [50, -1, 51, 52, 53, -1, 54, 55, 56, -1, 57, -1, 58, 59, 60, -1, 61, 62, 63, -1, 64],
                               [65, -1, -1, -1, -1, -1, -1, -1, -1, -1, 66, -1, -1, -1, -1, -1, -1, -1, -1, -1, 67],
                               [68, -1, 69, 70, 71, -1, 72, 73, 74, 75, 76, 77, 78, 79, 80, -1, 81, 82, 83, -1, 84],
                               [85, -1, 86, -1, 87, -1, 88, -1, 89, -1, -1, -1, 90, -1, 91, -1, 92, -1, 93, -1, 94],
                               [95, -1, 96, -1, 97, -1, 98, -1, 99, 100, 101, 102, 103, -1, 104, -1, 105, -1, 106, -1,
                                107],
                               [108, -1, 109, -1, 110, -1, 111, -1, -1, -1, -1, -1, -1, -1, 112, -1, 113, -1, 114, -1,
                                115],
                               [116, 117, 118, -1, 119, 120, 121, -1, -1, -1, -1, -1, -1, -1, 122, 123, 124, -1, 125,
                                126, 127]])
        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        self._forbidden_walls = forbidden_walls
        if forbidden_walls:
            self.__restrict_walls__()
        # Walls to insert
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))

        # Further options
        self._start_pos = kwargs.get('start_pos', None)
        if self._start_pos is not None:
            self.place_agent(self._start_pos)
        return


class OpenMaze(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays

    Open maze realizes a big open maze. I recommend the following setup:
        - a maze of shape [38 x 50]
        - the agent starts in state 1612 (coord: [32, 12])
        - small proximal reward in state 1405 (coord: [28, 5])
        - large distal reward in state 1644 (coord: [32, 44])
    This way the distal reward is EXACTLY 4 times far away (32) proximal reward (8), although this distance is measured
    by the 2-norm. In 1-norm the real difference between the start pos and the proximal site is 11.
    After the reward change:
        - place a single reward on state 278 (coord: [5, 28])
    This reward will be of a distance of 32 from the agent, and both the OG rewards (using a 2-norm). The 1-norm
    distance will be 43 between the starting site and the new reward, 43 between the and the old distal reward and the
    new reward; and 46 between the old proximal reward and the new reward.
    """

    def __init__(self, dim_x: int, dim_y: int, **kwargs):
        """
        Constructor of the double T maze class

        :param dim_x: the height of the maze
        :param dim_y: the width of the maze
        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        slip_prob       -- the probability of slipping after a step
                        start_pos       -- the initial position of the agent
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: stay, 1: up, 2: right, 3: down, 4: left}
        self._act = np.array(
            [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        self._maze = np.reshape(np.array(range(dim_x * dim_y)), (dim_x, dim_y))
        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        self._forbidden_walls = forbidden_walls
        if forbidden_walls:
            self.__restrict_walls__()
        # Walls to insert
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))

        # Further options
        self._start_pos = kwargs.get('start_pos', None)
        if self._start_pos is not None:
            self.place_agent(self._start_pos)
        return


class EDmaze(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays

    EDmaze stands for equidistant maze. In this maze, I recommend the following setup:
        - place large distal reward on state 94
        - place small proximal reward on state 87
        - place agent on state 102
    This way the distal reward takes EXACTLY 4 times as many steps (32) to reach as the proximal reward (8). After the
    reward change:
        - place a single reward on state 6
    This reward will be 32 steps away from the agent, the OG distal and the OG proximal rewards.
    """

    def __init__(self, **kwargs):
        """
        Constructor of the double T maze class

        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        slip_prob       -- the probability of slipping after a step
                        start_pos       -- the initial position of the agent
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: stay, 1: up, 2: right, 3: down, 4: left}
        self._act = np.array(
            [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        self._maze = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                               [19, -1, -1, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 21],
                               [22, 23, 24, 25, 26, -1, 27, 28, 29, 30, 31, 32, 33, -1, 34, 35, 36, 37, 38],
                               [-1, -1, -1, -1, 39, -1, -1, -1, -1, -1, -1, -1, 40, -1, 41, -1, -1, -1, -1],
                               [42, 43, 44, 45, 46, -1, 47, 48, 49, 50, 51, 52, 53, -1, 54, 55, 56, 57, 58],
                               [59, -1, -1, -1, -1, -1, 60, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 61],
                               [62, 63, 64, 65, 66, -1, 67, 68, 69, 70, 71, 72, 73, -1, 74, 75, 76, 77, 78],
                               [-1, -1, -1, -1, 79, -1, -1, -1, -1, -1, -1, -1, 80, -1, 81, -1, -1, -1, -1],
                               [82, 83, 84, 85, 86, -1, 87, 88, 89, 90, 91, 92, 93, -1, 94, 95, 96, 97, 98],
                               [99, -1, -1, -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 101],
                               [102, 103, 104, 105, 106, 107, 108, -1, 109, 110, 111, 112, 113, 114, 115, 116, 117, -1,
                                118],
                               [119, -1, -1, -1, -1, -1, -1, -1, 120, -1, -1, -1, -1, -1, -1, -1, 121, -1, 122],
                               [123, 124, 125, 126, 127, 128, 129, 130, 131, -1, -1, -1, -1, -1, -1, -1, 132, 133,
                                134]])
        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        self._forbidden_walls = forbidden_walls
        if forbidden_walls:
            self.__restrict_walls__()
        # Walls to insert
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))

        # Further options
        self._start_pos = kwargs.get('start_pos', None)
        if self._start_pos is not None:
            self.place_agent(self._start_pos)
        return


class SimpleMaze(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays
    SimpleMaze only contains 4 states, thus allowing for rapid experimentation.
    """

    def __init__(self, **kwargs):
        """
        Constructor of the double T maze class
        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        slip_prob       -- the probability of slipping after a step
                        start_pos       -- the initial position of the agent
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: up, 1: right, 2: down, 3: left}
        self._act = np.array(
            [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        self._maze = np.array([[0, 1, 2],
                               [3, 4, 5],
                               [6, 7, 8]])
        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        self._forbidden_walls = forbidden_walls
        if forbidden_walls:
            self.__restrict_walls__()
        # Walls to insert
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))

        # Further options
        self._start_pos = kwargs.get('start_pos', None)
        if self._start_pos is not None:
            self.place_agent(self._start_pos)
        return


class LinearMaze(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays
    LinearMaze is used for testing curiosity
    """

    def __init__(self, **kwargs):
        """
        Constructor of the double T maze class
        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        slip_prob       -- the probability of slipping after a step
                        start_pos       -- the initial position of the agent
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: up, 1: right, 2: down, 3: left}
        self._act = np.array(
            [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        self._maze = np.array([[0],
                               [1],
                               [2],
                               [3],
                               [4],
                               [5],
                               [6],
                               [7],
                               [8],
                               [9],
                               [10],
                               [11],
                               [12],
                               [13],
                               [14],
                               [15]])
        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        self._forbidden_walls = forbidden_walls
        if forbidden_walls:
            self.__restrict_walls__()
        # Walls to insert
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))

        # Further options
        self._start_pos = kwargs.get('start_pos', None)
        if self._start_pos is not None:
            self.place_agent(self._start_pos)
        return


class SmallEDmaze(Env):
    """
    A child class to env, where we can specify different mazes, without losing the functions already used in the parent
    class. The maze and all of its properties will have to be initialized as matrices contained in np arrays
    Small ED maze is a down scaled version of the ED maze.
    Proposed reward sites:
        The agent starts from state 3, each episode is 30 steps long
        Before reward change
            - small proximal on 22, distance is 6, value is 0.5 -- rr = 12
            - large distal on 34, distance is 12, value is 3 -- rr = 54
        After reward change
            - large distal on 27, distance is 12 (10 from each old reward), value is 3 -- rr = 54
    """

    def __init__(self, **kwargs):
        """
        Constructor of the double T maze class
        :param kwargs:  forbidden_walls -- is the agent allowed to choose to bump into the wall
                        slip_prob       -- the probability of slipping after a step
                        start_pos       -- the initial position of the agent
        """
        # Handling the potential kwargs
        forbidden_walls = kwargs.get('forbidden_walls', False)
        slip_prob = kwargs.get('slip_prob', 0)

        # Setting everything up so that we have a double-T maze
        Env.__init__(self)
        # The encoding of the possible actions: {0: up, 1: right, 2: down, 3: left}
        self._act = np.array(
            [np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])])
        # The maze itself
        self._maze = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                               [11, -1, -1, 12, -1, -1, -1, -1, -1, -1, 13],
                               [14, -1, -1, 15, 16, 17, 18, 19, 20, -1, 21],
                               [22, -1, -1, -1, -1, -1, -1, -1, 23, -1, 24],
                               [25, -1, -1, -1, 26, 27, 28, 29, 30, -1, 31],
                               [32, -1, -1, -1, 33, -1, -1, -1, -1, -1, 34],
                               [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]])
        # Transitions
        self._slip_prob = slip_prob
        # Where is the reward
        self._reward = np.zeros(self._maze.shape)
        # What is the likelihood of getting a reward
        self._reward_prob = np.zeros(self._maze.shape)
        # Where do we usually start from
        self._agent_pos = np.zeros(self._maze.shape)
        # Are there any forbidden actions (following the coding of _act)
        self._restrict = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))
        # Are the walls restricted
        self._forbidden_walls = forbidden_walls
        if forbidden_walls:
            self.__restrict_walls__()
        # Walls to insert
        self._walls = np.zeros((self._maze.shape[0], self._maze.shape[1], len(self._act)))

        # Further options
        self._start_pos = kwargs.get('start_pos', None)
        if self._start_pos is not None:
            self.place_agent(self._start_pos)
        return
