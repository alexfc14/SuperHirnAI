import time
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

def expected_survival_rate(guess, pool):
    codes_by_output = {}
    for i, secret in enumerate(pool):
        output = evaluate_guess(secret, guess, verbose=False)
        if output not in codes_by_output:
            codes_by_output[output] = 1            
        else:
            codes_by_output[output] += 1
    N = i + 1
    # compute expected value: sum_outputs (prob*x), prob = count/total, x=count/total
    weighed_mean_rate = sum([count/N*count/N for count in codes_by_output.values()])
    return weighed_mean_rate

def find_best_guess(openers, pool, time_limit=600, verbose=False):
    timeout = time.time() + time_limit
    min_rate = np.inf
    best_guess = None

    for i, guess in enumerate(openers):
        if time.time() > timeout:
            if verbose:
                print('reached time limit', time_limit, 'seconds')
            break
        rate = expected_survival_rate(guess, pool)
        if rate < min_rate:
            min_rate = rate
            best_guess = guess
            if verbose:
                print('new optimal', rate, guess, 'after', time.time()-(timeout-time_limit))
        else:
            if verbose == 2:
                print('not optimal', rate, guess)
    if verbose:
        print('explored', i+1, 'solutions:', min_rate, 'survivors')
    return best_guess

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

def representer_generator(n_colors, codelength):
    finish = np.arange(codelength)
    openers = itertools_generator(n_colors, codelength)
    for guess in openers:
        if is_representer(guess):
            yield guess
        if all(guess == finish):
            break

def random_code_generator(size, n_colors, codelength):
    for i in range(size):
        yield np.array([np.random.randint(n_colors) for i in range(codelength)])

def timeout(generator, time_limit, verbose=False):
    timeout = time.time() + time_limit
    for i, x in enumerate(generator):
        if time.time() > timeout:
            if verbose:
                print('reached time limit', time_limit, 'seconds', 'at item nº', i, ':', x)
            break
        yield x

def get_best_opener(n_colors, codelength):
    best_openers = {
        (8, 5): np.array([0, 0, 1, 2, 3])
    }
    default = np.arange(codelength)//2
    return best_openers.get((n_colors, codelength), default)

if __name__ == "__main__":

    n_colors = 8
    codelength = 5
    for experiment in range(10):
        openers = representer_generator(n_colors, codelength)
        pool = list(random_code_generator(500, n_colors, codelength))
        best_opener = find_best_guess(openers, pool, time_limit=6*3600, verbose=0)
        print('best guess', best_opener, 'rate', round(expected_survival_rate(best_opener, pool),3))


    n_colors = 8
    codelength = 5
    
    max_entropy = 0
    best_guess = None
    
    openers = representer_generator(n_colors, codelength)
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


    n_colors = 16
    codelength = 10

    openers = [
        np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),

        np.array((0, 0, 1, 2, 3, 4, 5, 6, 7, 8)),

        np.array((0, 0, 1, 1, 2, 3, 4, 5, 6, 7)),
        np.array((0, 0, 0, 1, 2, 3, 4, 5, 6, 7)),

        np.array((0, 0, 1, 1, 2, 2, 3, 4, 5, 6)),
        np.array((0, 0, 0, 1, 2, 2, 3, 4, 5, 6)),
        np.array((0, 0, 0, 1, 1, 2, 3, 4, 5, 6)),
        np.array((0, 0, 0, 0, 1, 2, 3, 4, 5, 6)),

        np.array((0, 0, 0, 0, 0, 1, 2, 3, 4, 5)),
        np.array((0, 0, 0, 0, 1, 1, 2, 3, 4, 5)),
        np.array((0, 0, 0, 1, 1, 1, 2, 3, 4, 5)),
        np.array((0, 0, 0, 1, 1, 2, 2, 3, 4, 5)),
        np.array((0, 0, 1, 1, 2, 2, 3, 3, 4, 5)),
        
        np.array((0, 0, 0, 0, 0, 0, 1, 2, 3, 4)),
        np.array((0, 0, 0, 0, 0, 1, 1, 2, 3, 4)),
        np.array((0, 0, 0, 0, 1, 1, 1, 2, 3, 4)),
        np.array((0, 0, 0, 0, 1, 1, 2, 2, 3, 4)),
        np.array((0, 0, 0, 1, 1, 1, 2, 2, 3, 4)),
        np.array((0, 0, 0, 1, 1, 1, 2, 2, 3, 4)),
        np.array((0, 0, 1, 1, 2, 2, 3, 3, 4, 4)),

        np.array((0, 0, 0, 0, 0, 0, 0, 1, 2, 3)),
        np.array((0, 0, 0, 0, 0, 0, 1, 1, 2, 3)),
        np.array((0, 0, 0, 0, 0, 1, 1, 2, 2, 3)),
        np.array((0, 0, 0, 0, 1, 1, 2, 2, 3, 3)),
        np.array((0, 0, 0, 1, 1, 1, 2, 2, 3, 3)),
        np.array((0, 0, 0, 1, 1, 1, 2, 2, 2, 3)),
        np.array((0, 0, 0, 0, 1, 1, 1, 1, 2, 3)),

        np.array((0, 0, 0, 0, 0, 0, 1, 1, 2, 2)),
        np.array((0, 0, 0, 0, 0, 1, 1, 1, 2, 2)),
        np.array((0, 0, 0, 0, 1, 1, 1, 2, 2, 2)),
        np.array((0, 0, 0, 0, 0, 0, 0, 1, 1, 2)),
        np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 2)),
        np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 1)),
        np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 1)),
        np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    ]
    pool = list(random_code_generator(2*10**5, n_colors, codelength))
    best_opener = find_best_guess(openers, pool, time_limit=6*3600, verbose=True)
    print('best guess', best_opener)

    n_colors = 13
    codelength = 8

    openers = representer_generator(n_colors, codelength)
    pool = list(random_code_generator(10**5, n_colors, codelength))
    best_opener = find_best_guess(openers, pool, time_limit=6*3600, verbose=True)
    print('best guess', best_opener)
