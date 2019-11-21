import numpy as np


class AcNoise(object):

    def reset(self):
        """Reset the noise process
        The method needs to be overriden noise processes, that generate
        temporally correlated noise across calls, but can be omitted when
        noise is generated from scratch every call.
        """
        pass


class NormalAcNoise(AcNoise):

    def __init__(self, mu, sigma):
        """Additive action space Gaussian noise"""
        self.mu = mu
        self.sigma = sigma

    def generate(self):
        """Generate Gaussian noise"""
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return "NormalAcNoise(mu={}, sigma={})".format(self.mu, self.sigma)


class OUAcNoise(AcNoise):

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        """Temporally correlated noise generated via an Orstein-Uhlenbeck process,
        well-suited for physics-based models involving inertia, such as locomotion.
        Implementation is based on the following post:
        http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def generate(self):
        """Generate noise based on statistics computed in previous generations,
        leveraging an Orstein-Uhlenbeck process to establish the temporal
        correlation accross the consecutive noise generations.
        """
        # Generate noise via the process
        noise = self.prev_noise
        noise += self.theta * (self.mu - self.prev_noise) * self.dt
        noise += self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        # Update the previous noise value to the current value for the next generation
        self.prev_noise = noise
        return noise

    def reset(self):
        self.prev_noise = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OUAcNoise(mu={}, sigma={})".format(self.mu, self.sigma)
