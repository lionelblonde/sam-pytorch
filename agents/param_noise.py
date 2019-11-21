class AdaptiveParamNoise(object):

    def __init__(self, initial_std=0.1, delta=0.1):
        """Adaptive parameter noise, as introduced in the paper
        'Parameter Space Noise for Exploration'
        Matthias Plappert, https://arxiv.org/abs/1706.01905

        Args:
            initial_std (float): Initial parameter noise standard deviation
            delta (float): Threshold used in the adaptive noise scaling heuristic
        """
        self.initial_std = initial_std
        self.delta = delta
        # Initialize the current standard deviation
        self.cur_std = initial_std

    def adapt_std(self, dist):
        """Adapt the parameter noise standard deviation based on distance `dist`
            `dist`: distance between the actions predicted respectively by the actor and
                    the adaptive-parameter-noise-perturbed actor (action space distance).
        Iteratively multiplying/dividing the standard deviation by this distance can be
        interpreted as adapting the scale of the parameter space noise over time and
        relating it the variance in action space that it induces.
        This heuristic is based on the Levenberg-Marquardt heuristic.

        A good choice of `delta` is the std of the desired action space additive normal noise,
        as it results in effective action space noise that has the same std as regular
        Gaussian action space noise (holds only because `dist` is an l2 distance in action space).
        (cf. section 'Adaptive noise scaling' in paper)
        """
        if dist < self.delta:  # Increase standard deviation
            self.cur_std *= 1.01
        else:  # Decrease standard deviation
            self.cur_std /= 1.01

    def adapt_delta(self, new_delta):
        """Adapt the threshold delta when following an eps-greedy heuristic"""
        self.delta = new_delta

    def __repr__(self):
        return "AdaptiveParamNoise(initial_std={}, delta={})".format(self.initial_std, self.delta)
