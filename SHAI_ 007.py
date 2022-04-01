import numpy as np
from Alice import evaluate_guess

from utils import is_compatible_with_history as is_compatible
from utils import get_guess, blacks, whites

class Cell:
    def __init__(self, n_colors, position=None, verbose=True):
        self.n_colors = n_colors
        self.position = position
        self.possible_values = list(range(n_colors))
        self.index = 0
        self.verbose = verbose
    
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
            if self.verbose:
                print('discard', value, 'at', self.position)
            # decrease the index if we deleted a value below current
            assert len(self.possible_values) > 1
            discard_index = self.possible_values.index(value)
            if discard_index <= self.index:
                self.index -= 1
            self.possible_values.remove(value)
    
    def set(self, value):
        self.index = self.possible_values.index(value)
    
    def confirm(self, value):
        if self.verbose:
            print('confirm', value, 'at', self.position)
        self.possible_values = [value]
        self.index = 0
    
    def unconfirmed(self):
        return len(self.possible_values) > 1

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
        self.cells = [Cell(n_colors, p, self.verbose) for p in range(codelength)]
    
    def current(self):
        return np.array([cell.current() for cell in self.cells])
    
    def unconfirmed(self):
        for cell in self.cells:
            if cell.unconfirmed():
                yield cell
    
    def absent_in_unconfirmed(self, value):
        for c in self.unconfirmed():
            if c.current() == value:
                return False
        return True
    
    def not_white_nor_black(self, value):
        if self.absent_in_unconfirmed(value):
            for cell in self.unconfirmed():
                cell.discard(value)

    def analyze(self, history):
        if len(history) == 0:
            return
        elif len(history) >= 1:
            last = history[-1]
            if blacks(last) == whites(last) == 0:
                # Discard all numbers present in guess from every cell
                for value in get_guess(last):
                    for cell in self.cells:
                        cell.discard(value)
            if blacks(last) + whites(last) == self.codelength:
                # Discard absent values
                possible = set(range(self.n_colors))
                present = set(get_guess(last))
                absent = possible - present
                for value in absent:
                    for cell in self.cells:
                        cell.discard(value)
        if len(history) < 2:
            return
        
        last = history[-1]
        past = history[-2]

        diff = (get_guess(last) != get_guess(past))
        slot = diff.tolist().index(True)
        if sum(diff) == 1:
            if blacks(last) != blacks(past):
                # Discard the one with highest blacks
                if blacks(last) > blacks(past):
                    value = get_guess(last)[slot]
                else:
                    value = get_guess(past)[slot]
                self.cells[slot].confirm(value)
            else:
                # Discard both
                for event in (last, past):
                    value = get_guess(event)[slot]
                    self.cells[slot].discard(value)
            if whites(last) == whites(past) and whites(last) == 0:
                if (blacks(last) == blacks(past)):
                    # Discard both everywhere
                    for event in (last, past):
                        value = get_guess(event)[slot]
                        self.not_white_nor_black(value)
                elif blacks(last) < blacks(past):
                    value = get_guess(last)[slot]
                    self.not_white_nor_black(value)
            elif whites(last) < whites(past) and blacks(last) == blacks(past):
                value = get_guess(last)[slot]
                self.not_white_nor_black(value)
            for event in history[:-1][::-1]:
                # Go back through all prev guesses of this cell that didn't contribute to white
                # TODO fix look back bug
                if whites(last) > whites(event) and blacks(last) == blacks(event):
                    diff = get_guess(last) != get_guess(event)
                    if sum(diff) == 1 and slot == diff.tolist().index(True)\
                    and whites(last) > whites(event):
                        value = get_guess(event)[slot]
                        # if self.verbose:
                            # print('realize', event, 'not whites in slot', slot, 'value', value)
                        self.not_white_nor_black(value)
        
    def set_current(self, values):
        assert len(values) == self.codelength
        for cell, value in zip(self.cells, values):
            cell.set(value)

    def random_compatible_guess(self, history, max_tries=100):
        for i in range(max_tries):
            guess = np.array([np.random.choice(c.possible_values) for c in self.cells])
            if is_compatible(guess, history):
                if self.verbose:
                    print('compatible')
                return guess

    # this method will be called by the Referee. have fun putting AI here:
    def make_a_guess(self, history, remaining_time_in_sec):
        # if self.verbose:
            # print(f"PLAYER: Remaining_time= {remaining_time_in_sec}")

        if not self.first_done:
            self.first_done = True
            for i in range(self.codelength):
                self.cells[i].set(self.best_opener[i])
            return self.current()
        
        self.analyze(history)

        
        guess = self.random_compatible_guess(history)
        if guess is not None:
            self.set_current(guess)
            return self.current()

        # Cells with fewer (>1) possible values offer more discarding % power
        prioritized_cells = sorted(
            self.unconfirmed(), 
            key = lambda cell: len(cell.possible_values)
        )
        if not len(prioritized_cells):
            return self.current()
        for cell in prioritized_cells:
            current_value = cell.current()
            # Values possible in fewer cells allow discarding via white
            # But for now we prioritize not currently appearing in other cells
            prioritized_values = sorted(
                [v for v in cell.possible_values if v != current_value],
                # key = lambda v:  len([c for c in self.cells if v in c.possible_values])
                key = lambda v:  len([c for c in prioritized_cells if v == c.current()])
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
    verbose = True
    for i in range(1000):
        alice = Alice(n_colors=n_colors,
            codelength=codelength,
            seed=np.random.randint(0, 2**31),
            verbose=verbose)
        player = Player(
            {'codelength':codelength, 'n_colors':n_colors}, 
            seed=np.random.randint(0, 2**31),
            verbose=verbose
        )
        while True:  # main game loop. loop till break because of won or lost game
            guess = player.make_a_guess(alice.history, None)
            alice_answer = alice(guess)
            # print('ALICE: Secret code is', alice.secret)
            if alice_answer == "GAME WON":
                break
        score = len(alice.history)
        print('score', score)
        print()