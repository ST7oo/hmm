import numpy as np


class HMM:
    """
    Hidden Markov Model Class

    Attributes:
        A: Transitions matrix.
        B: Emissions matrix.
        name_states: Labels of states.
        name_observations: Labels of observations.
        N: Number of possible states.
        M: Number of possible emissions.

    Public methods:
        gen_sequence(num_sequences)
        viterbi(observations)
        baum_welch(observations, max_iter)
    """

    def __init__(self, A, B, name_states, name_observations):
        """
        Constructor.

        Args:
            A (float[][]): Transitions matrix.
            B (float[][]): Emissions matrix.
            name_states (str[]): Labels of states.
            name_observations (str[]): Labels of observations.
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.name_states = name_states
        self.name_observations = name_observations
        self.N = len(name_states)
        self.M = len(name_observations)


    def gen_sequence(self, num_sequences=1):
        """
        Generates sequences of emissions given the current model.

        Args:
            num_sequences(int): Number of sequences to generate. Default 1.

        Returns:
            str[][]: Array with the sequences generated.
        """

        sequences = []
        for i in np.arange(num_sequences):
            states = []
            observations = []
            # get the initial state
            states.append(self._draw_from(self.A[0]))
            # get the initial emission
            observations.append(self._draw_from(self.B[states[0]]))
            # current time slice
            t = 1
            # get the current state given the previous state
            state = self._draw_from(self.A[states[t - 1]])
            # while state is not the final one
            while state < len(self.name_states) - 1:
                states.append(state)
                # get the emission
                observations.append(self._draw_from(self.B[state]))
                t += 1
                # get the next state
                state = self._draw_from(self.A[states[t - 1]])
            # get the labels of the emissions and append the sequence
            sequences.append([self.name_observations[i] for i in observations])
        return sequences


    def viterbi(self, observations):
        """
        Gets the most probable path using Viterbi algorithm.

        Args:
            observations (str[][]): List of sequences of observations.

        Returns:
            (str[], float)[]: List of sequences of states with their probability.
        """

        sequences = []
        # use log probabilities
        Alog = self._log_probabilities(self.A)
        Blog = self._log_probabilities(self.B)
        # for each sequence of observations
        for obs in observations:
            # number of time slices
            T = len(obs)
            # initialize
            delta = np.zeros([self.N, T])
            psi = np.zeros([self.N, T])
            # get the indices of the sequence of observations
            index_obs = self._index_observations(obs)
            # initial step
            delta[:, 0] = Alog[0, :] + Blog[:, index_obs[0]]
            psi[:, 0] = 0
            for t in np.arange(1, T):
                tmp = np.array([delta[:, t - 1]]).T + Alog
                # inductive step
                delta[:, t] = np.amax(tmp, axis=0) + Blog[:, index_obs[t]]
                psi[:, t] = np.argmax(tmp, axis=0)
            # normalize the probabilities
            sum_delta = np.exp(delta).sum(0)
            if sum_delta.all() > 0:
                delta_normalized = np.exp(delta) / sum_delta
            else:
                delta_normalized = np.exp(delta)
            # begin with the max probable state in the last time slice
            sequence = np.array([np.argmax(delta_normalized[:, T - 1])], dtype='int')
            # and with the max probability of last time slice
            prob = np.max(delta_normalized[:, T - 1])
            for t in np.arange(T - 1, 0, -1):
                # insert at the beginning of the sequence,
                # the state that gave most likely this state
                sequence = np.insert(sequence, 0, psi[sequence[0], t])
                # multiply by the probability of this state
                prob *= delta_normalized[sequence[1], t]
            # get the labels of the sequence
            seq = [self.name_states[i] for i in sequence]
            sequences.append((seq, prob))
        return sequences


    def baum_welch(self, observations, max_iter=20):
        """
        Trains the model using the Baum-Welch algorithm, given the observations.
        It updates itself.

        Args:
            observations (str[][]): List of sequences of observations.
            max_iter (int, optional): number of max iterations. Default 20.

        Returns:
            int: number of iterations used
        """

        # log likelihood of current model
        log_likelihood = float('-inf')
        # last log likelihood to check if it is improving
        last_log_likelihood = float('-inf')
        # iterations of training
        iter = 0
        # while it is improving (current log likelihood is greater than the last one)
        while iter < max_iter and (iter == 0 or log_likelihood > last_log_likelihood):
            last_log_likelihood = log_likelihood
            log_likelihood = 0
            # a-posterior probability numerator of A
            a_bar_num = np.zeros([self.N, self.N])
            # a-posterior probability denominator of A
            a_bar_den = np.zeros([self.N])
            # a-posterior probability numerator of B
            b_bar_num = np.zeros([self.N, self.M])
            # a-posterior probability denominator of B
            b_bar_den = np.zeros([self.N])
            # a-posterior probability of Pi
            pi_bar = np.zeros([self.N])

            # for each sequence of observations
            for obs in observations:
                # get alpha, beta and log likelihood of the sequence
                alpha, beta, log_prob_obs = self._forward_backward(obs)
                log_likelihood += log_prob_obs
                # number of time slices
                T = len(obs)
                # get the indices of the sequence of observations
                index_obs = self._index_observations(obs)
                # calculate gamma
                gamma_raw = alpha * beta
                gamma = (gamma_raw / gamma_raw.sum(0)).T
                # probability of starting in the state
                pi_bar += gamma[0, :]
                # number of times the state has been visited
                a_bar_den += gamma.sum(0)
                b_bar_den += gamma.sum(0)
                # calculate xi
                xi = np.zeros([self.N, self.N, T - 1])
                for t in np.arange(T - 1):
                    for i in np.arange(self.N):
                        xi[i, :, t] = alpha[i, t] * self.A[i, :] * self.B[:, index_obs[t + 1]] * beta[:, t + 1]
                # number of times the state fired
                a_bar_num += xi.sum(2)
                # number of times the state has been visited,
                # and its emission has been observed
                for k in np.arange(self.M):
                    indicator = np.array([self.name_observations[k] == x for x in obs])
                    b_bar_num[:, k] += gamma[indicator, :].sum(0)

            # update A
            A_bar = np.zeros([self.N, self.N])
            A_bar[0, :] = pi_bar / np.sum(pi_bar)
            for i in np.arange(1, self.N - 1):
                A_bar[i, :] = a_bar_num[i, :] / a_bar_den[i]
            self.A = A_bar

            # update B
            B_bar = np.zeros([self.N, self.M])
            for i in np.arange(1, self.N - 1):
                if b_bar_den[i] > 0:
                    B_bar[i, :] = b_bar_num[i, :] / b_bar_den[i]
                else:
                    B_bar[i, :] = b_bar_num[i, :]
            self.B = B_bar

            print(log_likelihood)
            iter += 1

        # correct final silent state
        self.A[self.N - 2, self.N - 1] = 1 - self.A[self.N - 2].sum()

        return iter


# Private methods

    def _forward_backward(self, observations):
        """Calculates alpha and beta using forward and backward algorithms."""

        # number of time slices
        T = len(observations)
        # scaling factor at each time
        factor = np.zeros([T])
        # initialize alpha and beta
        alpha = np.zeros([self.N, T])
        beta = np.zeros([self.N, T])
        # get the indices of the sequence of observations
        index_obs = self._index_observations(observations)

        # forward
        # initial step
        alpha[:, 0] = self.A[0, :] * self.B[:, index_obs[0]]
        factor[0] = 1.0 / np.sum(alpha[:, 0])
        alpha[:, 0] *= factor[0]
        # inductive step
        for t in np.arange(1, T):
            # the dot product takes care of multiplying and summing up
            alpha[:, t] = np.dot(alpha[:, t - 1], self.A) * self.B[:, index_obs[t]]
            factor[t] = 1.0 / np.sum(alpha[:, t])
            alpha[:, t] *= factor[t]

        # log likelihood of an observation sequence
        log_prob_obs = -(np.sum(np.log(factor)))

        # backward
        #initial step (1 * last factor)
        beta[:, T - 1] = factor[T - 1]
        for t in np.arange(T - 1, 0, -1):
            # the dot product takes care of multiplying and summing up
            beta[:, t - 1] = np.dot(self.A, self.B[:, index_obs[t]] * beta[:, t])
            beta[:, t - 1] *= factor[t - 1]

        return alpha, beta, log_prob_obs

    def _draw_from(self, probabilities):
        """Returns a random choice given an array of probabilities."""
        return np.random.choice(len(probabilities), 1, p=probabilities)[0]

    def _log_probabilities(self, X):
        """Returns the log of the values in X"""
        lp = np.zeros((X.shape))
        lp[X > 0] = np.log(X[X > 0])
        lp[X == 0] = float('-inf')
        return lp

    def _index_observations(self, obs):
        """Returns the indices of the observations passed"""
        y = []
        for o in obs:
            y.append(self.name_observations.index(o))
        return y


            # def forward(self, observations):
    #     # sequences = []
    #     # for i in np.arange(len(observations)):
    #     T = len(observations)
    #     c = np.zeros([T])
    #     alpha = np.zeros([self.N, T])
    #     index_obs = self._index_observations(observations)
    #     alpha[:, 0] = self.A[0, :] * self.B[:, index_obs[0]]
    #     c[0] = 1.0 / np.sum(alpha[:, 0])
    #     alpha[:, 0] *= c[0]
    #     for t in np.arange(1, T):
    #         alpha[:, t] = np.dot(
    #             alpha[:, t - 1], self.A) * self.B[:, index_obs[t]]
    #         c[t] = 1.0 / np.sum(alpha[:, t])
    #         alpha[:, t] *= c[t]
    #     log_prob_obs = -(np.sum(np.log(c)))
    #     return alpha, log_prob_obs, c
    #     # sequences.append((alpha, log_prob_obs, c))
    #     # return sequences

    # def backward(self, observations, c):
    #     # sequences = []
    #     # for i in np.arange(len(observations)):
    #     T = len(observations)
    #     index_obs = self._index_observations(observations)
    #     beta = np.zeros([self.N, T])
    #     beta[:, T - 1] = c[T - 1]
    #     for t in np.arange(T - 1, 0, -1):
    #         beta[:, t - 1] = np.dot(self.A,
    #                                 self.B[:, index_obs[t]] * beta[:, t])
    #         beta[:, t - 1] *= c[t - 1]
    #     return beta
    #     #     sequences.append(beta)
    #     # return sequences
