#!/home/goethe/venv/superhirn/bin/python3
import sys
import os
import numpy as np
import time
import argparse
import yaml
import pandas as pd
import importlib
import itertools
import hashlib
import sqlite3
import ray


import Referee


def get_seeds(ts):
    if ts["salt"] == "non-deterministic":
        salt = "".join([str(_) for _ in np.random.randint(0, 2**31, size=5)])
    else:
        salt = ts["salt"]
    seeds = [f'{ts["title"]}_{salt}_'+str(_) for _ in range(ts["n_games"])]
    seeds = [hashlib.sha224(_.encode()).hexdigest() for _ in seeds]
    seeds = [int(_, 16) for _ in seeds]
    # numpy digests single-seeds of 32 bits. We will need two per game. So 2^64
    seeds = [_ % (2**64) for _ in seeds]
    return(seeds)


def save_result_to_db(result, ts, ctrl):
    # Martin: This is bad coding stype, having this dependency between the output of Referee.play and this function. To be done better
    # convert to dataframe
    col_names = ["won-lost", "score", "duration",
                 "secret", "game-history", "seed", "player_name"]
    result = pd.DataFrame(result, columns=col_names)
    result["seed"] = result["seed"].astype(
        str)  # maybe too large for DB integer
    result["duration"] = np.round(result["duration"], 2)
    # I know: gamemodus is redundant! Just meant as cross-check.
    result["gamemodus"] = [ts["gamemodus"] for _ in range(len(result))]
    result = result[["gamemodus", "player_name", "score", "duration",
                     "won-lost", "secret", "game-history", "seed"]]  # just resort columns

    # save out to db
    con = sqlite3.connect(ctrl["pars"]["db_filepath"])
    result.to_sql(ts["title"], con, if_exists="append",
                  index=False)  # , dtype={"seed":"TEXT"})
    con.close()
    print(
        f"INFO: Results appended to table '{ts['title']}' of '{ctrl['pars']['db_filepath']}'")


def main(control_filepath="settings.yaml"):

    # load settings:
    print(
        f"INFO: Loading control settings from yaml file '{control_filepath}'")
    with open(control_filepath) as fp:
        ctrl = yaml.load(fp, Loader=yaml.FullLoader)

    # initializing ray cluster on localhost
    print(f"INFO: Setting up ray on localhost.")
    ray.init()  # include_dashboard=True)
    if ctrl["pars"]["n_games_in_parallel"] > ray.available_resources()["CPU"] / 2:
        print("Warning: If too many games are played in parallel, time controll will not work correctly. Buy a bigger machine, tight-arse!")

    # Iteration over tournament sections (ts), which are given as a list in yaml file.
    for ts in ctrl["tournament_sections"]:
        gpars = ctrl["gamemodi"][ts["gamemodus"]]  # game parameters
        participating_players = ts["players"]
        assert all([os.path.isfile(_+".py") for _ in participating_players]
                   ), "There are participants which I don't know. Python module required with same name."
        seeds_of_repetitions = get_seeds(ts)

        # define all games and chunk into chunks which will be played in parallel.
        chunk_size = ctrl["pars"]["n_games_in_parallel"]
        games_of_that_section = itertools.product(
            seeds_of_repetitions, participating_players)
        # chunking into chunks of size chunk_size
        game_rounds = iter(lambda: tuple(itertools.islice(
            games_of_that_section, chunk_size)), ())

        for game_round in game_rounds:
            print(
                f"\nINFO: Start of {len(game_round)} games of tournament_section {ts['title']}: {game_round}.")
            result = [Referee.play.remote(
                gpars, *seed_player, ctrl["pars"]["verbose"]) for seed_player in game_round]
            result = ray.get(result)
            assert game_round == tuple([tuple(
                X[-2:]) for X in result]), f"Input/output data is not properly aligned. Should actually not happen."

            save_result_to_db(result, ts, ctrl)


if __name__ == "__main__":

    infoTXT = "Superhirn tournament. Let the games begin!"
    parser = argparse.ArgumentParser(
        description=infoTXT, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--control_filepath", default="settings.yaml",
                        help="All tournament control via a yaml file at that location.")
    args = parser.parse_args()

    main(args.control_filepath)
