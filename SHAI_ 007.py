import numpy as np
from Alice import evaluate_guess

from utils import is_compatible_with_history as is_compatible
from utils import get_guess, blacks, whites

class Cell:
    def __init__(self, n_colors):
        self.n_colors = n_colors
        self.possible_values = list(range(n_colors))
        self.index = 0
    
    def current(self):
        return self.possible_values[self.index]
    
    def adjust_index(self):
        if self.index >= len(self.possible_values):
            self.index = 0
            return True
        return False
    
    def next(self):
        self.index += 1
        return self.adjust_index()
    
    def discard(self, value):
        if value in self.possible_values:
            self.possible_values.remove(value)
            self.adjust_index()
    
    def set(self, value):
        self.index = self.possible_values.index(value)
    
    def confirm(self, value):
        self.possible_values = [value]
        self.index = 0

class Player():

    def __init__(self, gpars, seed, verbose=False):

        self.n_colors = gpars["n_colors"]
        self.codelength = gpars["codelength"]
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        if self.verbose:
            print("PLAYER: Init player heuristics")

        self.first_done = False
        # self.best_opener = np.array([0, 0, 1, 1, 2])
        self.best_opener = self.rng.integers(0, self.n_colors, size=self.codelength)
        self.slot = self.codelength - 1
        self.cells = [Cell(n_colors) for i in range(codelength)]
    
    def current(self):
        return np.array([cell.current() for cell in self.cells])
    
    def analyze(self, history):
        if len(history) == 0:
            return
        elif len(history) >= 1:
            last = history[-1]
            if blacks(last) == whites(last) == 0:
                # Discard all numbers present in guess from every cell
                for v in get_guess(last):
                    for cell in self.cells:
                        cell.discard(v)
        if len(history) == 1:
            return
        
        last = history[-1]
        past = history[-2]

        diff = (get_guess(last) != get_guess(past))
        if sum(diff) == 1:
            if blacks(last) != blacks(past):
                # Discard the one with highest blacks
                slot = diff.tolist().index(True)
                if blacks(last) > blacks(past):
                    value = get_guess(last)[slot]
                else:
                    value = get_guess(past)[slot]
                self.cells[slot].confirm(value)
            else:
                # Discard both
                slot = diff.tolist().index(True)
                for event in (last, past):
                    value = get_guess(event)[slot]
                    self.cells[slot].discard(value)
            if whites(last) == whites(past):
                if (whites(last) == 0) and (blacks(last) <= blacks(past)):
                    # Discard both everywhere
                    slot = diff.tolist().index(True)
                    for event in (last, past):
                        value = get_guess(event)[slot]
                        for cell in self.cells:
                            cell.discard(value)
            elif whites(last) < whites(past):
                slot = diff.tolist().index(True)
                v = get_guess(last)[slot]
                unconfirmed = [c for c in self.cells if len(c.possible_values) > 1]
                # If it's not present in other unconfirmed cells, discard
                if len([c for c in unconfirmed if v == c.current()]) == 0:
                    for c in unconfirmed:
                        c.discard(v)

    # this method will be called by the Referee. have fun putting AI here:
    def make_a_guess(self, history, remaining_time_in_sec):
        if self.verbose:
            print(f"PLAYER: Remaining_time= {remaining_time_in_sec}")

        if not self.first_done:
            self.first_done = True
            for i in range(self.codelength):
                self.cells[i].set(self.best_opener[i])
            return self.current()
        
        self.analyze(history)

        # Cells with fewer (>1) possible values offer more discarding % power
        prioritized_cells = sorted(
            [c for c in self.cells if len(c.possible_values) > 1], 
            key = lambda cell: len(cell.possible_values)
        )
        for cell in prioritized_cells:
            current_value = cell.current()
            # Values possible in fewer cells allow discarding via white
            # But for now we prioritize not appearing in other cells
            prioritized_values = sorted(
                [v for v in cell.possible_values if v != current_value],
                # key = lambda v:  len([c for c in self.cells if v in c.possible_values])
                key = lambda v:  len([c for c in self.cells if v == c.current()])
            )
            for value in prioritized_values:
                cell.set(value)
                if is_compatible(self.current(), history):
                    return self.current()
            # If no value is found compatible for this cell, reset and move on
            cell.set(current_value)
        # If no code different by one cell is compatible, default to the first
        cell = prioritized_cells[0]
        value = sorted(
                [v for v in cell.possible_values if v != cell.current()],
                # key = lambda v:  len([c for c in self.cells if v in c.possible_values])
                key = lambda v:  len([c for c in self.cells if v == c.current()])
        )[0]
        cell.set(value)
        return self.current()


if __name__ == "__main__":
    from Alice import Alice
    n_colors = 13
    codelength = 8
    alice = Alice(n_colors=n_colors,
                  codelength=codelength,
                  seed=np.random.randint(0, 2**31),
                  verbose=True)
    player = Player(
        {'codelength':codelength, 'n_colors':n_colors}, 
        seed=np.random.randint(0, 2**31)
    )
    while True:  # main game loop. loop till break because of won or lost game
        guess = player.make_a_guess(alice.history, None)
        alice_answer = alice(guess)
        if alice_answer == "GAME WON":
            break
    score = len(alice.history)
    print('score', score)