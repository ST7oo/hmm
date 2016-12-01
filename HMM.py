import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter


class HMM:

    def __init__(self, A=None, B=None, name_states=None, name_observations=None):
        self.A = A
        self.B = B
        self.name_states = name_states
        self.name_observations = name_observations

    def gen_sequence(self, num_sequences=1):
        def draw_from(probabilities):
            return np.random.choice(len(probabilities), 1, p=probabilities)[0]
        sequences = []
        for i in np.arange(0, num_sequences):
            states = []
            observations = []
            states.append(draw_from(self.A[0].toarray()[0]))
            observations.append(draw_from(self.B[states[0]].toarray()[0]))
            t = 1
            state = draw_from(self.A[states[t - 1]].toarray()[0])
            while state < len(self.name_states) - 1:
                states.append(state)
                observations.append(draw_from(self.B[state].toarray()[0]))
                t += 1
                state = draw_from(self.A[states[t - 1]].toarray()[0])
            sequences.append(itemgetter(*observations)(self.name_observations))
        return sequences

    def viterbi(self, observations):
        delta = np.zeros([len(self.name_states), len(observations)])
        # delta[:, 0] = self.A[0] * self.B[]
        # states_it = np.arange(len(self.name_states) - 1)
        # V = np.zeros((len(observations), len(self.name_states)))
        # prev_states = np.zeros((len(observations), len(self.name_states)))
        # index_obs = self.name_observations.index(observations[0])
        # for i in states_it:
        #     V[0][i] = self.A[1][i] * self.B[i][index_obs]
        #     prev_states[0][i] = -1
        # for t in np.arange(1, len(observations)):
        #     index_obs = self.name_observations.index(observations[t])
        #     for i in states_it:
        #         max_trans_prob = np.max(V[t - 1][s] * self.A[s][i]
        #                                 for s in states_it)
        #         for s in states_it:
        #             if V[t - 1][s] * self.A[s][i] == max_trans_prob:
        #                 V[t][i] = max_trans_prob * self.A[i][index_obs]
        #                 prev_states[t][i] = s
        #                 break
        # return V

states = ['INIT', 'Onset', 'Mid', 'End', 'FINAL']
observations = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
A = csr_matrix([[0, 1, 0, 0, 0],
                [0, 0.3, 0.7, 0, 0],
                [0, 0, 0.9, 0.1, 0],
                [0, 0, 0, 0.4, 0.6]])
B = csr_matrix([[0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.2, 0.3, 0, 0, 0, 0],
                [0, 0, 0.2, 0.7, 0.1, 0, 0],
                [0, 0, 0, 0.1, 0, 0.5, 0.4]])
seq = [['C1', 'C2', 'C3', 'C4', 'C4', 'C6', 'C7'],
       ['C2', 'C2', 'C5', 'C4', 'C4', 'C6', 'C6']]
h = HMM(A, B, states, observations)
print(h.gen_sequence(3))
# print(h.viterbi(observations))
