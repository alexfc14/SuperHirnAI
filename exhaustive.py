import numpy as np
from generator import Generator, generator

from utils import find_best_guess, get_best_opener, is_compatible_with_history as is_compatible, itertools_generator, lpMin
from utils import get_guess, blacks, whites, col, row, find_best_guess

class Player():

    def __init__(self, gpars, seed, verbose=False):

        self.n_colors = gpars["n_colors"]
        self.codelength = gpars["codelength"]
        self.verbose = verbose
        if self.verbose:
            print("PLAYER: Init player heuristics")

        self.best_opener = get_best_opener(self.n_colors, self.codelength)
        self.pool = Generator(self.n_colors, self.codelength)
    
    def information(self):
        try:
            N = len(self.pool)
        except:
            N = self.n_colors**self.codelength
        return np.log2(N)
    
    def exhaustive_guess(self, history):
        pool = []
        for secret in self.pool:
            if is_compatible(secret, history):
                pool.append(secret.copy())
        self.pool = pool

        solution = find_best_guess(pool, pool, time_limit=1., verbose=self.verbose)
        return solution

    # this method will be called by the Referee. have fun putting AI here:
    def make_a_guess(self, history, remaining_time_in_sec):
        # if self.verbose:
        #     print(f"PLAYER: Remaining_time= {remaining_time_in_sec}")

        if not len(history):
            guess = self.best_opener      
        else:
            guess = self.exhaustive_guess(history)

        return guess


if __name__ == "__main__":
    from Alice import Alice
    n_colors = 13
    codelength = 8
    for i in range(100):
        seed=np.random.randint(0, 2**31)
        # seed=1325043143
        print('seed', seed)
        alice = Alice(n_colors=n_colors,
            codelength=codelength,
            seed=seed,
            verbose=False)
        player = Player(
            {'codelength':codelength, 'n_colors':n_colors}, 
            seed=1,
            # seed=np.random.randint(0, 2**31),
            verbose=False
        )
        print('secret', alice.secret, 'info', round(codelength*np.log2(n_colors),1))
        while True:  # main game loop. loop till break because of won or lost game
            guess = player.make_a_guess(alice.history, None)
            alice_answer = alice(guess)
            print('move nº', len(alice.history), 'info', round(player.information(), 2), 'bits', guess, alice_answer)
            if alice_answer == "GAME WON":
                break
        score = len(alice.history)
        info = codelength*np.log2(n_colors)
        print('score', score, 'info:', round(info,1), 'bits', 'rate', round(info/score, 2), 'bits/guess')