# A non-static rewrite of the PrivCountSketch.py class ()

# This is mainly used in simulations to test the PrivCountSketch frequency oracle
# The core logic for private count sketch [Charikar-Chen-Farach-Colton 2004]

import numpy as np
import math
import random

from pure_ldp.core import FreqOracleClient

class PCSClient(FreqOracleClient):
    def __init__(self, epsilon, l, w, hash_funcs):
        """
        Private Count Sketch (PCS) Algorithm
        Args:
            epsilon (float): Privacy Budget Epsilon
            l (integer): Number of hash functions for the sketch
            w (integer): Size of sketch  vector
            hash_funcs (list of funcs): A list of hash function mapping data to {0...m-1} (can be generated by CMSServer)
        """
        super().__init__(epsilon, None)
        self.l = l
        self.w = w
        self.epsilon = epsilon
        self.hash_funcs = hash_funcs
        self.bias = math.exp(self.epsilon) / (1 + math.exp(self.epsilon))

    def update_params(self, epsilon=None, d=None, index_mapper=None, l=None, w=None):
        super().update_params(epsilon, d, index_mapper)
        self.l = l if l is not None else self.l
        self.w = w if w is not None else self.w

        if epsilon is not None:
            self.bias = math.exp(self.epsilon) / (1 + math.exp(self.epsilon))

    # Duchi-Jordan-Wainwright 2013, Bassily-Smith-2015 randomizer
    def _perturb(self, data):
        h_loc = data[0]
        g_val = data[1]

        privatized_bit_vec = 2 * np.random.binomial(1, 0.5, self.w) - 1
        privatized_bit_vec[h_loc] = g_val * random.choices([1,-1], weights=[self.bias, 1-self.bias])[0]
        return privatized_bit_vec

    def privatise(self, data):
        """
        Privatises data item using CMS/HCMS

        Args:
            data: item to be privatised

        Returns: Privatised data

        """
        data = str(data)

        hash_id = np.random.randint(0, self.l)
        hash_pair = self.hash_funcs[hash_id]

        # Count sketch, h and g
        h_loc = hash_pair[0](data)
        g_val = 2*hash_pair[1](data)-1

        privatized_vec = self._perturb((h_loc, g_val))

        return privatized_vec, hash_id
