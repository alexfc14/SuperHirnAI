import numpy as np
from Alice import evaluate_guess


def is_compatible(code, history):
    for guess, observed_output in history:
        expected_output = evaluate_guess(code, guess, verbose=False)
        if observed_output != expected_output:
            return False
    return True


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
            print("PLAYER: Init player heuristics")

        self.guess_index = -1
        self.first_done = False
        self.best_opener = np.array([0, 0, 1, 1, 2])

    def next_guess(self):
        self.guess_index += 1

    # this method will be called by the Referee. have fun putting AI here:

    def make_a_guess(self, history, remaining_time_in_sec):

        if self.verbose:
            print(f"PLAYER: Remaining_time= {remaining_time_in_sec}")

        if not self.first_done:
            self.first_done = True
            return self.best_opener

        compatible = False
        while not compatible:
            guess = self.rng.integers(0, self.n_colors, size=self.codelength)
            compatible = is_compatible(guess, history)

        return guess
