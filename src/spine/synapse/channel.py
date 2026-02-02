import numpy as np

class SynapseInterface:

    def current_profile(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def update_activation(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def update_gate(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")
