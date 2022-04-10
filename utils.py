from collections import Counter
import pulp
import itertools
import numpy as np
from Alice import evaluate_guess
from generator import generator
from itertools import islice

def get_guess(event):
    return event[0]

def blacks(event):
    return event[1][0]

def whites(event):
    return event[1][1]


def col(X, v):
    if type(X) == np.ndarray:
        return X[:, v]
    else:
        return [X[i][v] for i in X.keys()]

def row(X, i):
    if type(X) == np.ndarray:
        return X[i]
    else:
        return [X[i][v] for v in X[i].keys()]

def lpMin(problem, x, x1, x2, name_suffix, M=10**10):
    """Enforce x = min(x1, x2) in LP using the big M trick."""
    problem += x <= x1
    problem += x <= x2
    y = pulp.LpVariable(name=f"y{name_suffix}", cat=pulp.LpBinary)
    problem += x >= x1 - M*(1-y)
    problem += x >= x2 - M*(y)


def itertools_generator(n_colors, codelength):
    for i in itertools.product(range(n_colors), repeat=codelength):
        yield np.array(i)

def is_compatible(secret, guess, guess_output):
    return guess_output == evaluate_guess(secret, guess, verbose=False)

def is_compatible_with_history(secret, history):
    return all(
        is_compatible(secret, guess, output)
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


def get_entropy_gain(guess, n_colors, codelength):
    Nc = n_colors**codelength
    secret_probability = 1/Nc
    prior_information = information(Nc, n_colors)
    entropy = 0
    codes_by_output = {}
    for secret in generator(n_colors, codelength):
        output = evaluate_guess(secret, guess, verbose=False)
        if output not in codes_by_output:
            codes_by_output[output] = 1
        else:
            codes_by_output[output] += 1
        Ns = codes_by_output[output]
        posterior_information = information(Ns, n_colors)
        information_gain = prior_information - posterior_information
        entropy += secret_probability * information_gain
    return entropy

def is_representer(guess):
    if guess[0] != 0:
        return False
    diff = guess[1:] - guess[:-1]
    if not all(0 <= diff):
        return False
    if not all(diff <= 1):
        return False
    
    # Smaller values should have higher value counts
    counts = np.array(list(Counter(guess).values()))
    diff = counts[1:] - counts[:-1]
    if not all(diff <= 0):
        return False
    
    return True

if __name__ == "__main__":
    n_colors = 10
    codelength = 8

    max_entropy = 0
    best_guess = None

    # openers = [
    #     np.array((0, 1, 2, 3, 4)),
    #     np.array((0, 0, 1, 2, 3)),
    #     np.array((0, 0, 1, 1, 2)),
    #     np.array((0, 0, 0, 1, 2)),
    #     np.array((0, 0, 0, 1, 1)),
    #     np.array((0, 0, 0, 0, 1)),
    #     np.array((0, 0, 0, 0, 0))
    # ]
    openers = itertools_generator(n_colors, codelength)

    for guess in openers:
        if not is_representer(guess):
            continue
        entropy = get_entropy_gain(guess, n_colors, codelength)
        if entropy > max_entropy:
            max_entropy = entropy
            best_guess = guess
        print('guess', guess, 'has entropy gain', entropy, 'or',
            n_colors ** (information(n_colors**codelength, n_colors) - entropy), 'average possible codes')

    print('best opener for a code of', n_colors, 'colors', 'and length',
        codelength,  'is', best_guess, 'with entropy', max_entropy)
