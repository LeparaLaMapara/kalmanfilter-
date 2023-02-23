import numpy as np

class KalmanFilter:
    def __init__(self, state_dim, obs_dim, A, B, H, Q, R, P):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = np.zeros((state_dim, 1))
    
    def predict(self, u=None):
        self.x = np.dot(self.A, self.x)
        if u is not None:
            self.x += np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
    
    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.state_dim)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def run_kalman_filter(time_series, A, B, H, Q, R, P, initial_state):
    state_dim = initial_state.shape[0]
    obs_dim = time_series.shape[1]
    kf = KalmanFilter(state_dim, obs_dim, A, B, H, Q, R, P)
    kf.x = initial_state
    
    filtered_states = np.zeros((len(time_series), state_dim))
    for i in range(len(time_series)):
        kf.predict()
        kf.update(time_series[i])
        filtered_states[i] = kf.x.T
        
    return filtered_states

# Define the Kalman filter parameters
state_dim = 2
obs_dim = 1
A = np.eye(state_dim)
B = np.zeros((state_dim, 1))
H = np.eye(obs_dim, state_dim)
Q = 0.01 * np.eye(state_dim)
R = 1.0 * np.eye(obs_dim)
P = np.eye(state_dim)

# Generate some time series data
t = np.linspace(0, 10, 1000)
x1 = np.sin(t)