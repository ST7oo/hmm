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
seq = [['C1', 'C2', 'C3', 'C4', 'C4', 'C6', 'C7'],
       ['C2', 'C2', 'C5', 'C4', 'C4', 'C6', 'C6'],
       ['C1', 'C2', 'C3', 'C6']]
h = HMM(A1, B1, states, observations)
# h = HMM(A, B, states, observations)
# print(h.gen_sequence(3))
print(h.viterbi(seq))
