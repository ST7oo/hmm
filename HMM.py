import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from operator import itemgetter


class HMM:

    def __init__(self, A=None, B=None, name_states=None, name_observations=None):
        self.A = A
        self.B = B
        self.name_states = name_states
        self.name_observations = name_observations
        self.N = len(name_states)
        self.M = len(name_observations)
        self.Alog = self.log_probabilities(self.A)
        self.Blog = self.log_probabilities(self.B)

    def log_probabilities(self, X):
        lp = np.zeros((X.shape))
        lp[X > 0] = np.log(X[X > 0])
        lp[X == 0] = float('-inf')
        return lp

    def gen_sequence(self, num_sequences=1):
        def draw_from(probabilities):
            return np.random.choice(len(probabilities), 1, p=probabilities)[0]
        sequences = []
        for i in np.arange(0, num_sequences):
            # states = []
            # observations = []
            # states.append(draw_from(self.A[0].toarray()[0]))
            # observations.append(draw_from(self.B[states[0]].toarray()[0]))
            # t = 1
            # state = draw_from(self.A[states[t - 1]].toarray()[0])
            # while state < len(self.name_states) - 1:
            #     states.append(state)
            #     observations.append(draw_from(self.B[state].toarray()[0]))
            #     t += 1
            #     state = draw_from(self.A[states[t - 1]].toarray()[0])
            # sequences.append(itemgetter(*observations)(self.name_observations))
            states = []
            observations = []
            states.append(draw_from(self.A[0]))
            observations.append(draw_from(self.B[states[0]]))
            t = 1
            state = draw_from(self.A[states[t - 1]])
            while state < len(self.name_states) - 1:
                states.append(state)
                observations.append(draw_from(self.B[state]))
                t += 1
                state = draw_from(self.A[states[t - 1]])
            sequences.append(itemgetter(*observations)(self.name_observations))
        return sequences

    def viterbi(self, observations):
        sequences = []
        for i in np.arange(len(observations)):
            T = len(observations[i])
            delta = np.zeros([self.N, T])
            psi = np.zeros([self.N, T])
            index_obs = self.name_observations.index(observations[i][0])
            delta[:, 0] = self.Alog[0, :] + self.Blog[:, index_obs]
            psi[:, 0] = 0
            for t in np.arange(1, T):
                tmp = np.array([delta[:, t - 1]]).T + self.Alog
                index_obs = self.name_observations.index(observations[i][t])
                delta[:, t] = np.amax(tmp, axis=0) + self.Blog[:, index_obs]
                psi[:, t] = np.argmax(tmp, axis=0)
            max_prob = np.exp(np.max(delta[:, T - 1]))
            sequence = np.array([np.argmax(delta[:, T - 1])], dtype='int')
            for t in np.arange(T - 1, 0, -1):
                sequence = np.insert(sequence, 0, psi[sequence[0], t])
            sequences.append((sequence, max_prob))
        return sequences

    def forward(self, observations):
        # sequences = []
        # for i in np.arange(len(observations)):
        T = len(observations)
        c = np.zeros([T])
        alpha = np.zeros([self.N, T])
        index_obs = self.name_observations.index(observations[0])
        alpha[:, 0] = self.A[0, :] * self.B[:, index_obs]
        c[0] = 1.0 / np.sum(alpha[:, 0])
        alpha[:, 0] *= c[0]
        for t in np.arange(1, T):
            index_obs = self.name_observations.index(observations[t])
            alpha[:, t] = np.dot(
                alpha[:, t - 1], self.A) * self.B[:, index_obs]
            c[t] = 1.0 / np.sum(alpha[:, t])
            alpha[:, t] *= c[t]
        log_prob_obs = -(np.sum(np.log(c)))
        return alpha, log_prob_obs, c
        # sequences.append((alpha, log_prob_obs, c))
        # return sequences

    def backward(self, observations, c):
        # sequences = []
        # for i in np.arange(len(observations)):
        T = len(observations)
        beta = np.zeros([self.N, T])
        beta[:, T - 1] = c[T - 1]
        for t in np.arange(T - 1, 0, -1):
            index_obs = self.name_observations.index(observations[t])
            beta[:, t - 1] = np.dot(self.A,
                                    self.B[:, index_obs] * beta[:, t])
            beta[:, t - 1] *= c[t - 1]
        return beta
        #     sequences.append(beta)
        # return sequences

    def bw(self, observations, max_iter=20):
        log_likelihoods = []
        for epoch in np.arange(max_iter):
            for obs in observations:
                alpha, log_prob_obs, c = self.forward(obs)
                beta = self.backward(obs, c)
                T = len(obs)
                for t in np.arange(T - 1):
                    index_obs = self.name_observations.index(obs[t + 1])

    def baum_welch(self, observations, max_iter=20):
        # best_A = self.A.copy()
        # best_B = self.B.copy()
        log_likelihoods = []
        # val_log_likelihoods = []
        for epoch in np.arange(max_iter):
            log_likelihood = 0
            b_bar_den = np.zeros([self.N])
            a_bar_den = np.zeros([self.N])
            # si_sj_all = np.zeros([self.N, self.N])
            a_bar_num = np.zeros([self.N, self.N])
            pi_bar = np.zeros([self.N])
            b_bar_num = np.zeros([self.N, self.M])
            for obs in observations:
                alpha, log_prob_obs, c = self.forward(obs)
                beta = self.backward(obs, c)
                # print(alpha)
                # print(beta)
                log_likelihood += log_prob_obs
                T = len(obs)
                w_k = 1.0 / -(log_prob_obs + np.log(T))
                gamma_raw = alpha * beta
                gamma = gamma_raw / gamma_raw.sum(0)
                pi_bar += w_k * gamma[:, 0]
                b_bar_den += w_k * gamma.sum(1)
                a_bar_den += w_k * gamma[:, :T - 1].sum(1)
                xi = np.zeros([self.N, self.N, T - 1])
                for t in np.arange(T - 1):
                    index_obs = self.name_observations.index(obs[t + 1])
                    for i in np.arange(self.N):
                        xi[i, :, t] = alpha[i, t] * self.A[i, :] * \
                            self.B[:, index_obs] * beta[:, t + 1]
                # si_sj_all += w_k * xi.sum(2)
                # a_bar_num += w_k * xi[:, :, :T - 1].sum(2)
                a_bar_num += w_k * xi.sum(2)
                B_bar = np.zeros([self.N, self.M])
                for k in np.arange(self.M):
                    indicator = np.array(
                        [self.name_observations[k] == x for x in obs])
                    B_bar[:, k] = gamma.T[indicator, :].sum(0)
                b_bar_num += w_k * B_bar
                # print(alpha)
                # print(beta)
                # print(xi)
                # print(a_bar_num)
                # print(a_bar_den)
            # update A
            A_bar = np.zeros([self.N, self.N])
            A_bar[0, :] = pi_bar / np.sum(pi_bar)
            for i in np.arange(1, self.N - 1):
                A_bar[i, :] = a_bar_num[i, :] / a_bar_den[i]
            self.A = A_bar
            # update B
            for i in np.arange(self.N):
                if b_bar_den[i] > 0:
                    b_bar_num[i, :] = b_bar_num[i, :] / b_bar_den[i]
                else:
                    b_bar_num[i, :] = b_bar_num[i, :]
            self.B = b_bar_num
            print(log_likelihood)
            log_likelihoods.append(log_likelihood)
            # print(self.A)
            if epoch > 1 and log_likelihoods[epoch - 1] <= log_likelihood:
                print('not improving')
                break
        return self


states = ['INIT', 'Onset', 'Mid', 'End', 'FINAL']
observations = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
A = csr_matrix([[0, 1, 0, 0, 0],
                [0, 0.3, 0.7, 0, 0],
                [0, 0, 0.9, 0.1, 0],
                [0, 0, 0, 0.4, 0.6]])
B = csr_matrix([[0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.2, 0.3, 0, 0, 0, 0],
                [0, 0, 0.2, 0.7, 0.1, 0, 0],
                [0, 0, 0, 0.1, 0, 0.5, 0.4],
                [0, 0, 0, 0, 0, 0, 0]])
A1 = np.array([[0, 1, 0, 0, 0],
               [0, 0.3, 0.7, 0, 0],
               [0, 0, 0.9, 0.1, 0],
               [0, 0, 0, 0.4, 0.6],
               [0, 0, 0, 0, 0]])
B1 = np.array([[0, 0, 0, 0, 0, 0, 0],
               [0.5, 0.2, 0.3, 0, 0, 0, 0],
               [0, 0, 0.2, 0.7, 0.1, 0, 0],
               [0, 0, 0, 0.1, 0, 0.5, 0.4],
               [0, 0, 0, 0, 0, 0, 0]])
A_ini = np.array([[0, 1, 0, 0, 0],
                  [0, 0.5, 0.5, 0, 0],
                  [0, 0, 0.5, 0.5, 0],
                  [0, 0, 0, 0.5, 0.5],
                  [0, 0, 0, 0, 0]])
B_ini = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0.33, 0.33, 0.33, 0, 0, 0, 0],
                  [0, 0, 0.33, 0.33, 0.33, 0, 0],
                  [0, 0, 0, 0.33, 0, 0.33, 0.33],
                  [0, 0, 0, 0, 0, 0, 0]])
seq = [['C1', 'C2', 'C3', 'C4', 'C4', 'C6', 'C7'],
       ['C2', 'C2', 'C5', 'C4', 'C4', 'C6', 'C6']]
seq1 = [['C1', 'C2', 'C3', 'C4', 'C4', 'C6', 'C7'],
        ['C2', 'C2', 'C5', 'C4', 'C4', 'C6', 'C6'],
        ['C1', 'C2', 'C3', 'C6']]
seq_train = [['C1', 'C4', 'C6', 'C7', 'C6'],
             ['C1', 'C5', 'C7'],
             ['C2', 'C1', 'C3', 'C1', 'C1', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4',
                 'C4', 'C5', 'C5', 'C3', 'C4', 'C4', 'C3', 'C3', 'C5', 'C7', 'C6', 'C7'],
             ['C2', 'C1', 'C5', 'C4', 'C4', 'C4', 'C4',
              'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C6'],
             ['C3', 'C4', 'C7', 'C7'],
             ['C2', 'C3', 'C4', 'C3', 'C4', 'C3', 'C6', 'C6'],
             ['C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4',
              'C4', 'C5', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C3', 'C6'],
             ['C3', 'C3', 'C1', 'C4', 'C3', 'C5',
                 'C4', 'C4', 'C4', 'C4', 'C4', 'C6'],
             ['C2', 'C3', 'C5', 'C4', 'C7'],
             ['C2', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C6'],
             ['C1', 'C4', 'C6', 'C7'],
             ['C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4',
              'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C3', 'C5', 'C7'],
             ['C2', 'C4', 'C7'],
             ['C3', 'C4', 'C7'],
             ['C2', 'C1', 'C4', 'C4', 'C4', 'C4', 'C3',
              'C4', 'C4', 'C4', 'C3', 'C4', 'C5', 'C7'],
             ['C1', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C6', 'C7'],
             ['C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C6'],
             ['C1', 'C3', 'C4', 'C4', 'C3', 'C5', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3',
              'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C7'],
             ['C1', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4',
              'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C6', 'C7'],
             ['C3', 'C4', 'C6'],
             ['C2', 'C4', 'C4', 'C5', 'C6'],
             ['C2', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4',
              'C3', 'C4', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C7'],
             ['C1', 'C5', 'C4', 'C4', 'C7'],
             ['C1', 'C4', 'C5', 'C4', 'C6'],
             ['C3', 'C4', 'C5', 'C4', 'C4', 'C3', 'C4', 'C5', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4',
              'C4', 'C4', 'C5', 'C4', 'C5', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C6'],
             ['C3', 'C3', 'C3', 'C4', 'C3', 'C5', 'C3', 'C4',
              'C5', 'C4', 'C4', 'C4', 'C4', 'C6', 'C4'],
             ['C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C6'],
             ['C3', 'C2', 'C1', 'C1', 'C2', 'C3', 'C3', 'C6', 'C6', 'C6'],
             ['C3', 'C4', 'C4', 'C4', 'C3', 'C6', 'C6'],
             ['C2', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C5',
              'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C3', 'C5', 'C3', 'C4', 'C3', 'C7', 'C6'],
             ['C1', 'C4', 'C5', 'C4', 'C6', 'C6'],
             ['C3', 'C1', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4',
              'C3', 'C3', 'C3', 'C4', 'C4', 'C4', 'C7', 'C4'],
             ['C1', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C7', 'C4'],
             ['C3', 'C1', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C7'],
             ['C2', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C5', 'C4', 'C5', 'C4', 'C4',
              'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C6'],
             ['C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C6'],
             ['C2', 'C1', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4',
              'C4', 'C3', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C3', 'C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4'],
             ['C1', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C3', 'C6', 'C6'],
             ['C1', 'C4', 'C5', 'C4'],
             ['C2', 'C1', 'C2', 'C3', 'C3', 'C4', 'C4', 'C5', 'C4', 'C3', 'C6'],
             ['C1', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4',
              'C4', 'C3', 'C4', 'C4', 'C5', 'C5', 'C3', 'C4', 'C7', 'C7'],
             ['C2', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C4',
              'C4', 'C4', 'C4', 'C4', 'C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4', 'C7', 'C6', 'C7'],
             ['C2', 'C3', 'C6'],
             ['C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4',
                 'C3', 'C4', 'C6', 'C6', 'C4', 'C7'],
             ['C2', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C7'],
             ['C3', 'C1', 'C1', 'C1', 'C2', 'C4', 'C4', 'C4',
              'C3', 'C4', 'C4', 'C6', 'C6', 'C6', 'C4', 'C7'],
             ['C3', 'C2', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C5', 'C4', 'C5', 'C3', 'C4',
              'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C3', 'C3', 'C6', 'C4'],
             ['C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C5', 'C4', 'C4', 'C5', 'C4', 'C4',
              'C3', 'C4', 'C3', 'C4', 'C3', 'C4', 'C3', 'C4', 'C3', 'C4', 'C7', 'C7', 'C7'],
             ['C1', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4',
                 'C4', 'C4', 'C5', 'C6', 'C6', 'C6'],
             ['C2', 'C4', 'C4', 'C4', 'C6', 'C7'],
             ['C3', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4',
              'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C7', 'C7', 'C6'],
             ['C3', 'C4', 'C4', 'C3', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C7'],
             ['C3', 'C4', 'C4', 'C6', 'C7'],
             ['C1', 'C4', 'C4', 'C4', 'C7', 'C6'],
             ['C3', 'C1', 'C3', 'C3', 'C4', 'C4', 'C4', 'C6'],
             ['C3', 'C1', 'C4', 'C3', 'C4', 'C4',
                 'C4', 'C4', 'C4', 'C3', 'C4', 'C6'],
             ['C3', 'C1', 'C1', 'C1', 'C1', 'C3', 'C7', 'C6'],
             ['C1', 'C4', 'C6', 'C6'],
             ['C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C3', 'C3', 'C6'],
             ['C1', 'C3', 'C5', 'C5', 'C4', 'C3', 'C4', 'C4', 'C4', 'C7'],
             ['C3', 'C3', 'C7', 'C7'],
             ['C1', 'C3', 'C3', 'C4', 'C3', 'C5',
                 'C5', 'C3', 'C4', 'C3', 'C4', 'C6'],
             ['C1', 'C3', 'C6', 'C6'],
             ['C1', 'C4', 'C5', 'C4', 'C4', 'C6', 'C6'],
             ['C2', 'C1', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C3', 'C4'],
             ['C3', 'C5', 'C4', 'C3', 'C3', 'C4', 'C4',
                 'C4', 'C3', 'C4', 'C5', 'C4', 'C7'],
             ['C3', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4',
              'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C7', 'C7'],
             ['C2', 'C1', 'C3', 'C3', 'C4', 'C3', 'C4',
                 'C3', 'C4', 'C4', 'C4', 'C4', 'C6'],
             ['C2', 'C1', 'C4', 'C5', 'C4', 'C4', 'C7', 'C4', 'C6', 'C6'],
             ['C1', 'C1', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C7'],
             ['C3', 'C1', 'C4', 'C5', 'C4', 'C6'],
             ['C1', 'C3', 'C4', 'C3', 'C5', 'C4', 'C3', 'C4',
              'C4', 'C4', 'C4', 'C3', 'C6', 'C6', 'C6'],
             ['C2', 'C1', 'C3', 'C1', 'C1', 'C3', 'C4', 'C4', 'C4', 'C6', 'C6'],
             ['C1', 'C2', 'C3', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3',
              'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C7', 'C6'],
             ['C3', 'C3', 'C4', 'C4', 'C6'],
             ['C1', 'C4', 'C4', 'C4', 'C5', 'C3', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4',
              'C3', 'C4', 'C4', 'C4', 'C4', 'C5', 'C3', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C6'],
             ['C1', 'C3', 'C4', 'C4', 'C6', 'C6', 'C7'],
             ['C2', 'C3', 'C4', 'C4', 'C4', 'C4', 'C7'],
             ['C1', 'C3', 'C4', 'C4', 'C6', 'C6'],
             ['C1', 'C3', 'C3', 'C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C5', 'C4',
              'C4', 'C5', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C3', 'C7'],
             ['C2', 'C1', 'C4', 'C3', 'C3', 'C4', 'C3', 'C4', 'C4', 'C4',
              'C3', 'C4', 'C4', 'C3', 'C4', 'C5', 'C4', 'C5', 'C6', 'C7'],
             ['C1', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4',
              'C4', 'C4', 'C3', 'C3', 'C5', 'C5', 'C5', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C7'],
             ['C1', 'C4', 'C4', 'C4', 'C5', 'C4', 'C4',
              'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C7'],
             ['C2', 'C4', 'C5', 'C4', 'C4', 'C4', 'C4', 'C4', 'C3', 'C4',
              'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C7', 'C6'],
             ['C2', 'C4', 'C3', 'C5', 'C4', 'C3', 'C4', 'C4', 'C4', 'C3', 'C4', 'C4',
              'C4', 'C4', 'C4', 'C4', 'C4', 'C5', 'C5', 'C4', 'C3', 'C3', 'C3', 'C6'],
             ['C1', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4',
              'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C6', 'C6'],
             ['C3', 'C3', 'C4', 'C4', 'C4', 'C4',
                 'C5', 'C4', 'C4', 'C4', 'C5', 'C7'],
             ['C3', 'C4', 'C4'],
             ['C1', 'C4', 'C4', 'C7'],
             ['C1', 'C3', 'C4', 'C5', 'C3', 'C4', 'C7', 'C6'],
             ['C3', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C7', 'C4', 'C7'],
             ['C1', 'C3', 'C4', 'C4'],
             ['C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C6'],
             ['C3', 'C3', 'C5', 'C4', 'C5', 'C4'],
             ['C1', 'C1', 'C4', 'C4', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4',
              'C4', 'C4', 'C3', 'C4', 'C6', 'C7', 'C6', 'C6', 'C7'],
             ['C3', 'C3', 'C3', 'C4', 'C4', 'C7', 'C6'],
             ['C3', 'C1', 'C3', 'C4', 'C4', 'C3', 'C3', 'C6'],
             ['C2', 'C3', 'C3', 'C4', 'C4', 'C5', 'C6', 'C7'],
             ['C2', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4', 'C4',
              'C4', 'C4', 'C4', 'C3', 'C4', 'C4', 'C3', 'C7'],
             ['C1', 'C3', 'C2', 'C4', 'C4', 'C6', 'C7', 'C7']]
# h = HMM(A1, B1, states, observations)
# h = HMM(A, B, states, observations)
# print(h.gen_sequence(3))
# print(h.viterbi(seq))
# print(h.forward(seq))
h1 = HMM(A_ini, B_ini, states, observations)
h2 = h1.baum_welch(seq_train)
print('A', h2.A)
print('B', h2.B)
