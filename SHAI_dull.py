#!/home/goethe/venv/superhirn/bin/python3

import os, sys
import numpy as np

class Player():

    def __init__(self, gpars, seed, verbose=False):

        self.n_colors = gpars["n_colors"]
        self.codelength = gpars["codelength"]
        # other gpars parameters (such as max_n_moves...) are available but ignored by this Player.

        # Each Player gets its own random number generator to assure reproducibility. 
        # Please use that generator if you need random numbers.
        # If we would use np.random, then result depends on execution order of processes what is not reproducible.
        self.rng = np.random.default_rng(seed) 

        self.verbose = verbose
        if self.verbose:
            print ("PLAYER: Init player SHAI_dull")



    # this method will be called by the Referee. have fun putting AI here:
    def make_a_guess(self, history, remaining_time_in_sec):

        if self.verbose:
            print (f"PLAYER: Remaining_time= {remaining_time_in_sec}")

        # not very bright this "AI":
        guess = self.rng.integers(0, self.n_colors, size=self.codelength)
        return(guess)
