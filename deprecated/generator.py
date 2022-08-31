import numpy as np


class Generator:
    def __init__(self, n_colors, codelength):
        self.n_colors = n_colors
        self.codelength = codelength
        self.slot = self.codelength - 1
        self.current = np.zeros(codelength)

    def next(self):
        self.current[self.slot] += 1
        if self.current[self.slot] == self.n_colors:
            self.current[self.slot] = 0
            self.slot -= 1
            self.next()
        else:
            self.slot = self.codelength - 1

    def done(self):
        return all(self.current == self.n_colors - 1)

    def __iter__(self):
        while True:
            yield self.current
            if self.done():
                break
            self.next()

def generator(n_colors, codelength):
    slot = codelength - 1
    current = np.zeros(codelength)
    
    while True:
        yield current

        if all(current == n_colors - 1):
            break
        
        current[slot] += 1
        while current[slot] == n_colors:
            current[slot] = 0
            slot -= 1
            current[slot] += 1
        slot = codelength - 1


if __name__ == "__main__":
    n_colors = 3
    codelength = 3

    # for guess in Generator(n_colors, codelength):
    for guess in generator(n_colors, codelength):
        a = 0
        print(guess)
