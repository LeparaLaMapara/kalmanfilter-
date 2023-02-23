import torch
import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, state_dim, obs_dim, transition_model, observation_model):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.particles = None
        self.weights = None

    def initialize(self, init_state):
        self.particles = init_state.repeat(self.num_particles, 1)
        self.weights = torch.ones(self.num_particles) / self.num_particles

    def predict(self):
        noise = torch.randn_like(self.particles)
        self.particles = self.transition_model(self.particles) + noise

    def update(self, obs):
        obs = obs.repeat(self.num_particles, 1)
        likelihood = self.observation_model(obs, self.particles)
        self.weights *= likelihood
        self.weights /= self.weights.sum()

    def resample(self):
        indices = np.random.choice(self.num_particles, size=self.num_particles, replace=True, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = torch.ones(self.num_particles) / self.num_particles

    def step(self, obs):
        self.predict()
        self.update(obs)
        if 1 / (self.weights ** 2).sum() < self.num_particles / 2:
            self.resample()
        estimate = torch.sum(self.particles * self.weights.unsqueeze(-1), axis=0)
        return estimate