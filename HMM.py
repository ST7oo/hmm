import numpy as np
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

states = ['INIT', 'Onset', 'Mid', 'End', 'FINAL']
observations = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
A = [[0, 1, 0, 0, 0],
     [0, 0.3, 0.7, 0, 0],
     [0, 0, 0.9, 0.1, 0],
     [0, 0, 0, 0.4, 0.6]]
B = [[0, 0, 0, 0, 0, 0, 0],
     [0.5, 0.2, 0.3, 0, 0, 0, 0],
     [0, 0, 0.2, 0.7, 0.1, 0, 0],
     [0, 0, 0, 0.1, 0, 0.5, 0.4]]
seq = [['C1', 'C2', 'C3', 'C4', 'C4', 'C6', 'C7'],
       ['C2', 'C2', 'C5', 'C4', 'C4', 'C6', 'C6']]
h = HMM(A, B, states, observations)
print(h.gen_sequence())
