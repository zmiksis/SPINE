import numpy as np

class synapse():
    def __init__(self, release_pattern=None, release_profile=None):
        self.node = []
        self.type = 'constant'
        self.duration = 1.e-3
        self.decay_constant = 1.e-2
        self.j = 1.e-11
        self.t = -1.e99
        self.domain = 'cytosol'

        self._release_pattern = release_pattern or self.default_release_pattern
        self._release_profile = release_profile or self.default_release_profile

    def release_profile(self, *args, **kwargs):
        return self._release_profile(self, *args, **kwargs)

    def default_release_profile(self, *args, **kwargs):

        if self.type == 'constant' and self.t >= 0.:
            if self.t < self.duration:
                return self.j
            else:
                return 0.0

        if self.type == 'linear' and self.t >= 0.:
            if self.t < self.duration:
                return self.j * (1 - self.t / self.duration)
            else:
                return 0.0
            
        elif self.type == 'exponential' and self.t >= 0.:
            return self.j * np.exp(-self.t / self.decay_constant)
        
        else: return 0.0

    def release_pattern(self, *args, **kwargs):
        return self._release_pattern(self, *args, **kwargs)

    def default_release_pattern(self, *args, **kwargs):
        t = kwargs.get('t', None)
        self.t = t % 0.1 # 10 Hz stimulation