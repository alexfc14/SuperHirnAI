#!/home/goethe/venv/superhirn/bin/python3
import sys
import os
import numpy as np
import time
import yaml
import json
import pandas as pd
import importlib
import itertools

import ray
from ray.exceptions import GetTimeoutError

from Alice import Alice


@ray.remote
def play(gpars, seed, player_name, verbose=True):

    # defining seed for alice and player
    if seed is None:  # note: if called by current tournament_main.py, never None.
        alice_seed, player_seed = [
            int(X) for X in np.random.randint(0, 2**32, size=2)]
    else:
        # splitting 64-bit seed into two 32-bit ones. This impedes seed fishing attacks since the salt is set at competition day
        alice_seed = seed % (2**32)
        player_seed = seed // (2**32)

    # get Alice
    alice = Alice(n_colors=gpars["n_colors"],
                  codelength=gpars["codelength"],
                  seed=alice_seed,
                  verbose=verbose)

    # get a Player
    # dynamically import of the module where the Player class is defined
    Player_module = importlib.import_module(player_name)
    Player_actor_class = ray.remote(num_cpus=1)(
        Player_module.Player)  # wrap it to actor class
    # instantiation as actor on separate cpu since num_cpus==1
    player = Player_actor_class.remote(gpars, player_seed, verbose=verbose)

    starttime = time.time()
    while True:  # main game loop. loop till break because of won or lost game

        # remaining time
        total_time = gpars["main_time"] + \
            len(alice.history) * gpars["time_increment"]
        remaining_time = total_time - (time.time() - starttime)
        if remaining_time <= 0:
            game_result = "LOST: time's over"
            break

        # player's turn
        guess = player.make_a_guess.remote(alice.history, remaining_time)
        try:
            guess = ray.get(guess, timeout=remaining_time)
        except GetTimeoutError:
            game_result = "LOST: time's over"
            break
        except Exception as exception_handler:
            game_result = f"LOST: player caused exception {str(exception_handler)}"
            break

        # alice' turn
        try:
            alice_answer = alice(guess)
        except AssertionError:
            game_result = "LOST: invalid guess"
            break

        # jump out if game is won
        if alice_answer == "GAME WON":
            game_result = "WON"
            break

        # lost if max length exceeded
        if len(alice.history) >= gpars["max_n_moves"]:
            game_result = "LOST: max gamelength reached"
            break

    # Returning the game results
    duration = time.time() - starttime

    if game_result.startswith("WON"):
        # score is number of moves, if the game was won, otherwise max_n_moves
        score = len(alice.history)
    else:
        assert game_result.startswith("LOST")
        score = gpars["max_n_moves"]
    # convert secret and history to json strings. Nothing can be numpy anymore
    json_secret = json.dumps([int(X) for X in alice.secret])
    json_hist = [([int(Y) for Y in X[0]], X[1]) for X in alice.history]
    json_hist = json.dumps(json_hist)

    # we send back seed and player name to cross-check that results are aligned. This is not necessary.
    # ray also takes care of proper alignment. just to convince the participants, that results are not messed up.
    return([game_result, score, duration, json_secret, json_hist, seed, player_name])
