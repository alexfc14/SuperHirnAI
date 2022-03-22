import numpy as np
from Alice import evaluate_guess

from utils import is_compatible_with_history as is_compatible

class Cell:
    def __init__(self, n_colors):
        self.n_colors = n_colors
        self.possible_values = list(range(n_colors))
        self.index = 0
    
    def current(self):
        return self.possible_values[self.index]
    
    def next(self) -> bool:
        self.index += 1
        reached_celing = self.index == len(self.possible_values)
        if reached_celing:
            self.index = 0
        return reached_celing
    
    def discard(self, value):
        self.possible_values.remove(value)

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
        self.cells = [Cell(n_colors) for i in range(codelength)]
    
    def current(self):
        return np.array([cell.current() for cell in self.cells])
    
    def next(self):
        if self.cells[self.slot].next():
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
            # return self.best_opener
            return self.rng.integers(0, self.n_colors, size=self.codelength)

        while not is_compatible(self.current(), history):
            self.next()

        return self.current()

if __name__ == "__main__":
    from Alice import Alice
    n_colors = 13
    codelength = 5
    alice = Alice(n_colors=n_colors,
                  codelength=codelength,
                  seed=np.random.randint(0, 2**31),
                  verbose=True)
    player = Player({'codelength':codelength, 'n_colors':n_colors}, seed=np.random.randint(0, 2**31))
    while True:  # main game loop. loop till break because of won or lost game
        guess = player.make_a_guess(alice.history, None)
        alice_answer = alice(guess)
        if alice_answer == "GAME WON":
            break
    score = len(alice.history)
    print('score', score)