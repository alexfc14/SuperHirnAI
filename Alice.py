#!/home/goethe/venv/superhirn/bin/python3
import numpy as np
from collections import Counter


def evaluate_guess(secret, guess, verbose=True):
    c1 = secret.copy()
    c2 = guess.copy()
    # number black evaluation pins, one pin per correct color at correct position.
    correct = (c1 == c2)
    blacks = int(correct.sum())
    # drop those elements of code and guess which were fully correct and gave a black pin already
    c1 = c1[~correct]
    c2 = c2[~correct]
    # number white evaluation pins, i.e. number of color matches between c1,c2 after dropping the fully correct pairings
    wrong_position = list((Counter(c1) & Counter(c2)).elements())
    whites = len(wrong_position)
    if verbose:
        print(f"ALICE: Received guess {guess} gives score ({blacks},{whites})")
    return((blacks, whites))


class Alice():
    def __init__(self, n_colors=8, codelength=5, seed=None, verbose=True):
        self.n_colors = n_colors
        self.codelength = codelength
        self.verbose = verbose
        self.history = []

        if seed is not None:
            self.alice_seed = seed
        else:
            self.alice_seed = np.random.randint(0, 2**32, size=1)[0]

        if self.verbose:
            print("ALICE: Setting Alice's seed to", self.alice_seed)
        # create an independent rng to assure reproducability
        self.rng = np.random.default_rng(self.alice_seed)

        self.secret = self.rng.integers(0, self.n_colors, size=self.codelength)
        if self.verbose:
            print("ALICE: Secret code is", self.secret)

    def _convert_and_test_input(self, guess):
        guess = np.array(guess)
        assert guess.shape == (
            self.codelength,), f"Provided guess does not have the right codelength of {self.codelength}."
        assert np.all(guess == np.rint(guess)
                      ), f"Guess only contains integers."
        assert np.all((guess >= 0) & (guess < self.n_colors)
                      ), f"Only integers allowd with 0 <= integer < {self.n_colors}."
        if len(self.history) > 0:
            assert self.history[-1][1] != (self.codelength,
                                           0), f"You've won the game already, dumbhead!"
        return (guess)

    def __call__(self, guess):
        guess = self._convert_and_test_input(guess)
        score = evaluate_guess(self.secret, guess, verbose=self.verbose)
        self.history.append((guess, score))
        if score == (self.codelength, 0):
            return ("GAME WON")
        else:
            return(score)


if __name__ == "__main__":
    a = Alice(n_colors=8, codelength=5, seed=42, verbose=True)
    ret = a([6, 2, 3, 4, 7])
    print("Alice gave me the answer:", ret)
    ret = a([0, 6, 5, 3, 3])
    print("Alice gave me the answer:", ret)
    print("Game history so far is", a.history)
