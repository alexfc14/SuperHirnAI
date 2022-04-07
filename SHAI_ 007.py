from itertools import chain
import numpy as np
from Alice import evaluate_guess

from utils import is_compatible_with_history as is_compatible, lpMin
from utils import get_guess, blacks, whites, col, row

import pulp

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
        if self.unconfirmed():
            if self.verbose:
                print('confirm', value, 'at', self.position)
        elif self.current() != value:
            print('what??')

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
    
    def information(self):
        return sum([np.log2(len(c.possible_values)) for c in self.cells])
    
    def not_white_nor_black(self, slot, event):
        guess = get_guess(event)
        value = guess[slot]
        
        others = [guess[i] for i in range(self.codelength)
                    if slot != i and\
                    ( self.cells[i].unconfirmed() #confirmed cells do not contribute to whites in current guesses, but blacks
                    or guess[i] != self.cells[i].possible_values[0]) # but could if reanalizing when we had not confirmed this cell yet
                ]
        if value not in others:# or other whites in unconfirmed cells whatsoever
            for cell in self.unconfirmed():
                cell.discard(value)
    
    def not_black(self, slot, event):
        value = get_guess(event)[slot]
        self.cells[slot].discard(value)
    
    def black(self, slot, event):
        value = get_guess(event)[slot]
        self.cells[slot].confirm(value)

    def analyze(self, history):
        if len(history) == 0:
            return
        elif len(history) >= 1:
            last = history[-1]
            if blacks(last) == 0:
                for slot in range(self.codelength):
                    self.not_black(slot, last)
            if blacks(last) == self.codelength - len(list(self.unconfirmed())) and\
            all(self.current() == get_guess(last)):
                for c in self.unconfirmed():
                    slot = self.cells.index(c)
                    self.not_black(slot, last)
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
                # Confirm the one with highest blacks
                if blacks(last) > blacks(past):
                    self.black(slot, last)
                else:
                    self.black(slot, past)
            else:
                # Discard both
                for event in (last, past):
                    self.not_black(slot, event)
            
            if whites(last) == whites(past) and whites(last) == 0:
                if (blacks(last) == blacks(past)):
                    for event in (last, past):
                        self.not_white_nor_black(slot, event)
                elif blacks(last) < blacks(past):
                    self.not_white_nor_black(slot, last)
            elif whites(last) < whites(past) and blacks(last) == blacks(past):
                self.not_white_nor_black(slot, last)
            for event in history[:-1][::-1]:
                # Go back through all prev guesses of this cell that didn't contribute to white
                if whites(last) > whites(event) and blacks(last) == blacks(event):
                    diff = get_guess(last) != get_guess(event)
                    if sum(diff) == 1:
                        diff_slot = diff.tolist().index(True)
                        if slot == diff_slot:
                            self.not_white_nor_black(slot, event)
        # if self.verbose:
        #     print('reanalize the past at', history[:-1][-1])
        # self.analyze(history[:-1])

    def set_current(self, values):
        assert len(values) == self.codelength
        for cell, value in zip(self.cells, values):
            cell.set(value)

    def random_compatible_guess(self, history, max_tries=100):
        for i in range(max_tries):
            guess = np.array([
                c.possible_values[self.rng.integers(0, len(c.possible_values))]
                for c in self.cells])
            if is_compatible(guess, history):
                if self.verbose:
                    print('found compatible random guess')
                return guess

    def linear_programming_compatible_guess(self, history):
        Cells = range(self.codelength)
        Vals = range(self.n_colors)

        problem = pulp.LpProblem("SuperHirn Problem", pulp.LpMaximize)
        X = pulp.LpVariable.dicts(name="X", indices=(Cells, Vals), lowBound=0, upBound=1, cat=pulp.LpInteger)
        for i in Cells:
            problem += pulp.lpSum(row(X, i)) == 1, f"One value per cell {i}"
            for v in Vals:
                if v not in self.cells[i].possible_values:
                    problem += X[i][v] == 0, f"Remove discarded value {v} from cell {i}"
        
        for episode, event in enumerate(history):
            # Build the guess matrix with indicator vectors as rows
            G = np.zeros((self.codelength, self.n_colors))
            for i, v in enumerate(get_guess(event)):
                G[i, v] = 1
            
            # matches of each value mv is the minimum of number of occurrences of value v in guess g and in secret code x
            m = pulp.LpVariable.dicts(f"m({episode})", Vals, 0, self.codelength, pulp.LpInteger)
            for v in Vals:
                lpMin(problem, x=m[v], x1 = sum(col(G, v)), x2 = sum(col(X, v)), M=self.n_colors*100, name_suffix=f"({episode})_{v}")
                
            # Black matches
            problem += pulp.lpSum([pulp.lpDot(row(G, i), row(X, i)) for i in Cells]) == blacks(event), f"Blacks == {blacks(event)} at event {episode}: {event}"
            # White matches
            problem += pulp.lpSum(m) == blacks(event) + whites(event), f"Whites == {whites(event)} at event {episode}: {event}"
        
        # Warm start with last guess
        warm_start = len(history) > 0
        if warm_start:
            for c in Cells:
                for v in Vals:
                    X[c][v].setInitialValue(G[c][v])

        # solver = pulp.GUROBI_CMD(msg=0)
        solver=pulp.PULP_CBC_CMD(msg=0, warmStart=True, presolve=True)
        problem.solve(solver=solver)
        solution = np.zeros(self.codelength, dtype='int')
        for i in Cells:
            for v in Vals:
                if pulp.value(X[i][v]) == 1:
                    solution[i] = v
        compatible = is_compatible(solution, history)
        if self.verbose:
            print('status', pulp.LpStatus[problem.status])
        if not compatible:  
            print('no bueno')
        else:
            return solution


    def prioritized_guess(self, history):
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

    # this method will be called by the Referee. have fun putting AI here:
    def make_a_guess(self, history, remaining_time_in_sec):
        if self.verbose:
            print(f"PLAYER: Remaining_time= {remaining_time_in_sec}")

        self.analyze(history)

        guess = None

        # if not len(history):
        #     guess = self.best_opener      
        # if guess is None:
        #     if len(history) < 15:
        #         guess = self.random_compatible_guess(history, max_tries=10000)
        if guess is None:
            guess = self.linear_programming_compatible_guess(history)
        # if guess is None:
        #     guess = self.prioritized_guess(history)
        
        self.set_current(guess)
        return self.current()


if __name__ == "__main__":
    from Alice import Alice
    n_colors = 13
    codelength = 8
    verbose = False
    for i in range(1000):
        seed=np.random.randint(0, 2**31)
        # seed=1201655299
        # print(seed)
        alice = Alice(n_colors=n_colors,
            codelength=codelength,
            seed=seed,
            verbose=verbose)
        player = Player(
            {'codelength':codelength, 'n_colors':n_colors}, 
            seed=1,
            # seed=np.random.randint(0, 2**31),
            verbose=False
        )
        while True:  # main game loop. loop till break because of won or lost game
            guess = player.make_a_guess(alice.history, None)
            alice_answer = alice(guess)
            print('move nº', len(alice.history), 'info', player.information(), 'bits')
            if alice_answer == "GAME WON":
                break
        score = len(alice.history)
        print('score', score)