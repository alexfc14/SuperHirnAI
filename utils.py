
import itertools
import numpy as np
from Alice import evaluate_guess
from generator import generator

def get_guess(event):
    return event[0]

def blacks(event):
    return event[1][0]

def whites(event):
    return event[1][1]


def col(X, c):
    if type(X) == np.ndarray:
        return X[:, c]
    else:
        return [X[v][c] for v in range(len(X.keys()))]


def itertools_generator(n_colors, codelength):
    for i in itertools.product(range(n_colors), repeat=codelength):
        yield np.array(i)



def is_compatible(candidate, guess, guess_output):
    return guess_output == evaluate_guess(candidate, guess, verbose=False)

def is_compatible_with_history(candidate, history):
    return all(
        is_compatible(candidate, guess, output)
        for guess, output in history#[::-1]
    )

def survivors(secret, guess, candidates):
    output = evaluate_guess(secret, guess, verbose=False)
    for candidate in candidates:
        if is_compatible(candidate, guess, output):
            yield candidate


def count_survivors(secret, guess, candidates):
    survivor_count = 0
    for survivor in survivors(secret, guess, candidates):
        survivor_count += 1
    return survivor_count


def information(n_codes, n_colors):
    return np.log2(n_codes) / np.log2(n_colors)


def get_entropy(guess, n_colors, codelength):
    Nc = n_colors**codelength
    secret_probability = 1/Nc
    prior_information = information(Nc, n_colors)
    entropy = 0
    for secret in generator(n_colors, codelength):
        Ns = count_survivors(
            secret,
            guess,
            candidates=generator(n_colors, codelength)
        )
        posterior_information = information(Ns, n_colors)
        information_gain = prior_information - posterior_information
        entropy += secret_probability * information_gain
    return entropy


if __name__ == "__main__":
    n_colors = 4
    codelength = 5

    max_entropy = 0
    best_guess = None

    # openers = itertools_generator(n_colors, codelength)
    openers = [
        np.array((0, 1, 2, 3, 4)),
        np.array((0, 0, 1, 2, 3)),
        np.array((0, 0, 1, 1, 2)),
        np.array((0, 0, 0, 1, 2)),
        np.array((0, 0, 0, 1, 1)),
        np.array((0, 0, 0, 0, 1)),
        np.array((0, 0, 0, 0, 0))
    ]

    for guess in openers:
        entropy = get_entropy(guess, n_colors, codelength)
        if entropy > max_entropy:
            max_entropy = entropy
            best_guess = guess
        print('guess', guess, 'has entropy gain', entropy, 'or',
            n_colors ** (information(n_colors**codelength, n_colors) - entropy), 'average possible codes')

    print('best opener for a code of', n_colors, 'colors', 'and length',
        codelength,  'is', best_guess, 'with entropy', max_entropy)
