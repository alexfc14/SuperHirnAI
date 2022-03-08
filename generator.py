import numpy as np


class Generator:
    def __init__(self, n_colors, codelength):
        self.n_colors = n_colors
        self.codelength = codelength
        self.slot = self.codelength - 1
        self.current = np.zeros(codelength)

    def next(self):
        self.current[self.slot] += 1
        self.clip_units()

    def clip_units(self):
        if self.current[self.slot] == self.n_colors:
            self.current[self.slot] = 0
            self.slot -= 1
            self.next()
        else:
            self.slot = self.codelength - 1

    def done(self):
        for i in self.current:
            if i < self.n_colors - 1:
                return False
        return True

    def __iter__(self):
        done = False
        while not done:
            yield self.current
            done = self.done()
            self.next()


if __name__ == "__main__":
    n_colors = 3
    codelength = 3

    gen = Generator(n_colors, codelength)
    for guess in gen:
        a = 0
        print(guess)
