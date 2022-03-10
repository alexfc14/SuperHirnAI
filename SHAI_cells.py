import numpy as np
from Alice import evaluate_guess

from utils import is_compatible_with_history as is_compatible


class Player():

    def __init__(self, gpars, seed, verbose=False):

        self.n_colors = gpars["n_colors"]
        self.codelength = gpars["codelength"]
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        if self.verbose:
            print("PLAYER: Init player heuristics")


        self.first_done = False
        self.best_opener = np.array([0, 0, 1, 1, 2])
        self.slot = self.codelength - 1
        self.current = np.zeros(self.codelength, dtype='int')

    def next(self):
        self.current[self.slot] += 1
        if self.current[self.slot] == self.n_colors:
            self.current[self.slot] = 0
            self.slot -= 1
            self.next()
        else:
            self.slot = self.codelength - 1

    # this method will be called by the Referee. have fun putting AI here:
    def make_a_guess(self, history, remaining_time_in_sec):
        if self.verbose:
            print(f"PLAYER: Remaining_time= {remaining_time_in_sec}")

        if not self.first_done:
            self.first_done = True
            return self.best_opener

        while not is_compatible(self.current, history):
            self.next()

        return self.current

if __name__ == "__main__":
    from Alice import Alice
    n_colors=8
    codelength=5
    seed = np.random.randint(0, 2**31)
    alice = Alice(n_colors=n_colors,
                  codelength=codelength,
                  seed=seed,
                  verbose=True)
    player = Player({'codelength':codelength, 'n_colors':n_colors}, seed=seed)
    while True:  # main game loop. loop till break because of won or lost game
        guess = player.make_a_guess(alice.history, None)
        alice_answer = alice(guess)
        if alice_answer == "GAME WON":
            break
    score = len(alice.history)
    print('score', score)