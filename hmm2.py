import numpy as np

name_states = ['INIT','Onset','Mid','End','FINAL']
name_observations = ['C1','C2','C3','C4','C5','C6','C7']
A = np.array([[0,1,0,0,0],
    [0,0.3,0.7,0,0],
    [0,0,0.9,0.1,0],
    [0,0,0,0.4,0.6],
    [0,0,0,0,0]])
B = np.array([[0,0,0,0,0,0,0],
    [0.5,0.2,0.3,0,0,0,0],
    [0,0,0.2,0.7,0.1,0,0],
    [0,0,0,0.1,0,0.5,0.4],
     [0,0,0,0,0,0,0]])
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
seq_train = [['C1','C2','C3','C4','C4','C6','C7'],
      ['C2','C2','C5','C4','C4','C6','C6']]

A2 = A_ini.copy()
B2 = B_ini.copy()
N = len(name_states)
M = len(name_observations)


observations = seq_train[0]
T = len(observations)
# FORWARD
alpha = np.zeros([N,T])
# initial step
yt = name_observations.index(observations[0])
for i in np.arange(N):
    alpha[i,0] = A2[0,i] * B2[i,yt]
# inductive step
for t in np.arange(T-1):
    yt = name_observations.index(observations[t+1])
    for i in np.arange(N):
        s = 0
        for k in np.arange(N):
            s += alpha[k,t] * A2[k,i]
        alpha[i,t+1] = s * B2[i,yt]
prob_forward = 0
for i in np.arange(N):
    prob_forward += alpha[i,T-1] * A2[i,N-1]

# BACKWARD
beta = np.zeros([N,T])
# initial step
for i in np.arange(N):
    beta[i,T-1] = 1.0
#     beta[i,T-1] = A2[i,N-1]
# inductive step
for t in np.arange(T-2,-1,-1):
    yt = name_observations.index(observations[t+1])
    for i in np.arange(N):
        s = 0
        for k in np.arange(N):
            s += beta[k,t+1] * A2[i,k] * B2[k,yt]
        beta[i,t] = s
prob_backward = 0
yt = name_observations.index(observations[0])
for i in np.arange(N):
    prob_backward += A2[0,i] * B2[i,yt] * beta[i,0]


gamma = np.zeros([N,T])
for t in np.arange(T):
    denominator = 0
    for k in np.arange(N):
        denominator += alpha[k,t] * beta[k,t]
    for i in np.arange(N):
        gamma[i,t] = (alpha[i,t] * beta[i,t]) / denominator
xi = np.zeros([N,N,T])
for t in np.arange(T-1):
    denominator = 0
    yt = name_observations.index(observations[t+1])
    for i in np.arange(N):
        for k in np.arange(N):
            denominator += alpha[i,t] * A2[i,k] * beta[k,t+1] * B2[k,yt]
    for i in np.arange(N):
        for k in np.arange(N):
            xi[i,k,t] = (alpha[i,t] * A2[i,k] * beta[k,t+1] * B2[k,yt]) / denominator


# UPDATE A
for i in np.arange(N):
    A2[0,i] = gamma[i,0]
for i in np.arange(1,N-1):
    for k in np.arange(1,N):
        numerator = 0
        denominator = 0
        for t in np.arange(T):
            numerator += xi[i,k,t]
            denominator += gamma[i,t]
        A2[i,k] = numerator / denominator

# UPDATE B
for i in np.arange(1,N-1):
    for vk in np.arange(M):
        numerator = 0
        denominator = 0
        for t in np.arange(T):
            yt = name_observations.index(observations[t])
            if yt == vk:
                numerator += gamma[i,t]
            denominator += gamma[i,t]
        B2[i,vk] = numerator / denominator