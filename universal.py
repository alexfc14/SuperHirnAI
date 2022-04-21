import numpy as np

from utils import find_best_guess, is_compatible_with_history as is_compatible
from utils import get_guess, blacks, whites, col, row

import pulp
import gurobipy

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
        # self.best_opener = self.rng.integers(0, self.n_colors, size=self.codelength)
        self.slot = self.codelength - 1
        self.cells = [Cell(self.n_colors, p, self.verbose) for p in range(self.codelength)]

        self.value_counts = {}
    
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
        prior_information = self.information()

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
            # Constant guess tells us how many occurrences a value has
            if all(get_guess(last) == get_guess(last)[0]):
                v = get_guess(last)[0]
                self.value_counts[v] = blacks(last) + whites(last)
            # Avoid redundant guess of last color, make a special case
            v = self.n_colors - 1
            if len(history) == v and v not in self.value_counts and len(self.value_counts) == v:
                self.value_counts[v] = self.codelength - sum(self.value_counts.values())
                if self.value_counts[v] == 0:
                    for cell in self.cells:
                        cell.discard(v)
            # Discard one-count-values in ohter cells after confirming in one
            for v, count in self.value_counts.items():
                confirmed_cells = [c for c in self.cells if c.possible_values == [v]]
                if count == len(confirmed_cells) > 0:
                    for c in self.cells:
                         if c not in confirmed_cells:
                             c.discard(v)

        if len(history) >= 2:
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

            while self.information() < prior_information:
                self.analyze(history)
                prior_information = self.information()

    def lp_analyze(self, history):
        tail = 2
        if len(history) > self.n_colors + tail:
            Cells = range(self.codelength)
            for c in Cells:
                unique_values = len(set([get_guess(e)[c] for e in history[-tail:]]))
                if unique_values > 1:
                    continue
                cell = self.cells[c]
                if cell.unconfirmed():
                    v = get_guess(history[-1])[c]
                    if v in cell.possible_values:
                        self.lp_confirm()(history, v, c)
        tail = 2
        if len(history) > self.n_colors + tail:
            Cells = range(self.codelength)
            for c in Cells:
                cell = self.cells[c]
                for v in cell.possible_values:
                    self.lp_discard(history, v, c)

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
            for v in Vals:
                if v not in self.cells[i].possible_values:
                    X[i][v] = 0
            if not self.cells[i].unconfirmed():
                v = self.cells[i].possible_values[0]
                X[i][v] = 1
        for i in Cells:
            problem += pulp.lpSum(row(X, i)) == 1, f"One value per cell {i}"
        
        for episode, event in enumerate(history):
            # Build the guess matrix with indicator vectors as rows
            G = np.zeros((self.codelength, self.n_colors))
            for i, v in enumerate(get_guess(event)):
                G[i, v] = 1
            
            # Black matches
            problem += pulp.lpSum([pulp.lpDot(row(G, i), row(X, i)) for i in Cells]) == blacks(event), f"Blacks == {blacks(event)} at event {episode}: {event}"
        
        # Warm start with last guess
        warm_start = len(history) > 0
        if warm_start:
            for c in Cells:
                for v in Vals:
                    if type(X[c][v]) == pulp.LpVariable:
                        X[c][v].setInitialValue(G[c][v])
        
        # try:
        #     solver = pulp.GUROBI(msg=0, warmStart=True, timeLimit=15)
        #     problem.solve(solver=solver)
        # except Exception as e:
        #     if self.verbose:
        #         print('GUROBI failed, using CBC. Reason:', e.message)
        #     solver=pulp.PULP_CBC_CMD(msg=0, warmStart=True, presolve=True, timeLimit=10)
        #     problem.solve(solver=solver)
        
        MAX_SOLS = 200
        TEST_SURVIVE = 200
        solver = pulp.GUROBI(msg=0, warmStart=True, PoolSolutions=MAX_SOLS, PoolSearchMode=2, gapRel=0, timeLimit=30)
        problem.solve(solver=solver)

        model = problem.solverModel
        if self.verbose:
            print(model.SolCount, 'solutions found')
        pool = set()
        for solution_number in range(model.SolCount):
            # Iterate through feasible solutions
            model.setParam('SolutionNumber', solution_number)
            sol_vals = model.getAttr(gurobipy.GRB.Attr.Xn)

            for var, val in zip(problem._variables, sol_vals):
                var.varValue = val
            
            solution = np.zeros(self.codelength, dtype='int')
            for c in Cells:
                for v in Vals:
                    if pulp.value(X[c][v]) == 1:
                        solution[c] = v
            compatible = is_compatible(solution, history)
            if not compatible:
                # if self.verbose:
                #     print('incorrect solution from GUROBI!', solution)
                continue
            solution = tuple(solution)
            if solution in pool:
                # if self.verbose:
                #     print('duplicate solution from GUROBI!', solution)
                continue
            pool.add(solution)
        if self.verbose:
            print(len(pool), 'clean solutions')
        # Choose best solution
        pool = [np.array(x) for x in pool]
        if len(pool) == 0:
            if self.verbose:
                print('no solutions found!')
                print('status', pulp.LpStatus[problem.status])
            self.lp_analyze(history)
        else:
            solution = find_best_guess(pool, pool[:TEST_SURVIVE], 5, self.verbose)
            return solution
    
    def lp_discard(self, history, value, cell):
        # Formulate the general problem
        Cells = range(self.codelength)
        Vals = range(self.n_colors)

        problem = pulp.LpProblem("SuperHirn Problem", pulp.LpMaximize)
        X = pulp.LpVariable.dicts(name="X", indices=(Cells, Vals), lowBound=0, upBound=1, cat=pulp.LpInteger)
        for i in Cells:
            for v in Vals:
                if v not in self.cells[i].possible_values:
                    X[i][v] = 0
            if not self.cells[i].unconfirmed():
                v = self.cells[i].possible_values[0]
                X[i][v] = 1
        
        # Add constraint of cell having this value. If infeasible, discard value.
        X[cell][value] = 1

        for i in Cells:
            problem += pulp.lpSum(row(X, i)) == 1, f"One value per cell {i}"
        
        for episode, event in enumerate(history):
            # Build the guess matrix with indicator vectors as rows
            G = np.zeros((self.codelength, self.n_colors))
            for i, v in enumerate(get_guess(event)):
                G[i, v] = 1
            
            # Black matches
            problem += pulp.lpSum([pulp.lpDot(row(G, i), row(X, i)) for i in Cells]) == blacks(event), f"Blacks == {blacks(event)} at event {episode}: {event}"

        solver = pulp.GUROBI(msg=0, warmStart=True, timeLimit=15)
        problem.solve(solver=solver)

        solution = np.zeros(self.codelength, dtype='int')
        for c in Cells:
            for v in Vals:
                if pulp.value(X[c][v]) == 1:
                    solution[c] = v
        
        if not is_compatible(solution, history):
            self.cells[c].discard(value) 

    def lp_confirm(self, history, value, cell):
        # Formulate the general problem
        Cells = range(self.codelength)
        Vals = range(self.n_colors)

        problem = pulp.LpProblem("SuperHirn Problem", pulp.LpMaximize)
        X = pulp.LpVariable.dicts(name="X", indices=(Cells, Vals), lowBound=0, upBound=1, cat=pulp.LpInteger)
        for i in Cells:
            for v in Vals:
                if v not in self.cells[i].possible_values:
                    X[i][v] = 0
            if not self.cells[i].unconfirmed():
                v = self.cells[i].possible_values[0]
                X[i][v] = 1
        
        # Add constraint of cell not having this value. If infeasible, confirm value.
        X[cell][value] = 0

        for i in Cells:
            problem += pulp.lpSum(row(X, i)) == 1, f"One value per cell {i}"
        
        for episode, event in enumerate(history):
            # Build the guess matrix with indicator vectors as rows
            G = np.zeros((self.codelength, self.n_colors))
            for i, v in enumerate(get_guess(event)):
                G[i, v] = 1
            
            # Black matches
            problem += pulp.lpSum([pulp.lpDot(row(G, i), row(X, i)) for i in Cells]) == blacks(event), f"Blacks == {blacks(event)} at event {episode}: {event}"

        solver = pulp.GUROBI(msg=0, warmStart=True, timeLimit=15)
        problem.solve(solver=solver)

        solution = np.zeros(self.codelength, dtype='int')
        for c in Cells:
            for v in Vals:
                if pulp.value(X[c][v]) == 1:
                    solution[c] = v
        
        if not is_compatible(solution, history):
            self.cells[c].confirm(value)

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
            prioritized_values = sorted(
                [v for v in cell.possible_values if v != current_value],
                key = lambda v:  len([c for c in self.cells if v in c.possible_values])
                # key = lambda v:  len([c for c in prioritized_cells if v == c.current()])
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

    def constant_guess(self, v):
        return np.repeat(v, self.codelength)

    # this method will be called by the Referee. have fun putting AI here:
    def make_a_guess(self, history, remaining_time_in_sec):
        # if self.verbose:
        #     print(f"PLAYER: Remaining_time= {remaining_time_in_sec}")

        self.analyze(history)

        guess = None

        # if not len(history):
        #     guess = self.best_opener      
        # if guess is None:
        #     if len(history) < 15:
        #         guess = self.random_compatible_guess(history, max_tries=10000)
        if guess is None:
            if len(history) < self.n_colors - 1:
                if self.verbose:
                    print('constant', len(history))
                guess = self.constant_guess(len(history))
        if guess is None:
            if self.verbose:
                print('linear programming')
            guess = self.linear_programming_compatible_guess(history)
        if guess is None:
            if self.verbose:
                print('deductive guess')
            guess = self.prioritized_guess(history)

        self.set_current(guess)
        return self.current()


if __name__ == "__main__":
    from Alice import Alice
    n_colors = 72
    codelength = 45
    for i in range(1):
        # seed=np.random.randint(0, 2**31)
        seed=739163279
        # print(seed)
        alice = Alice(n_colors=n_colors,
            codelength=codelength,
            seed=seed,
            verbose=False)
        player = Player(
            {'codelength':codelength, 'n_colors':n_colors}, 
            seed=1,
            # seed=np.random.randint(0, 2**31),
            verbose=True
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