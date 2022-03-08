#!/home/goethe/venv/superhirn/bin/python3
import sys
import os
import yaml
import numpy as np
import argparse
import pandas as pd
import sqlite3

from scipy.stats import mannwhitneyu as u_test


def get_tablenames(ctrl):  # table name is title of tournament section
    con = sqlite3.connect(ctrl["pars"]["db_filepath"])
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [X[0] for X in cursor.fetchall()]
    print(
        f"INFO: DB '{ctrl['pars']['db_filepath']}' has data for tournament sections: {table_names}")
    return(table_names)


def load_results_from_db(ctrl, ts_title):

    # load from db
    con = sqlite3.connect(ctrl["pars"]["db_filepath"])
    df = pd.read_sql(f"select * from {ts_title}", con)
    con.close()
    print(
        f"INFO: DataFrame of shape {df.shape} loaded from table '{ts_title}' of '{ctrl['pars']['db_filepath']}'")
    return(df)


def err(X):
    return np.std(X)/np.sqrt(len(X))


def main(control_filepath="settings.yaml"):

    # load settings:
    print(
        f"INFO: Loading control settings from yaml file '{control_filepath}'")
    with open(control_filepath) as fp:
        ctrl = yaml.load(fp, Loader=yaml.FullLoader)

    ts_titles_in_db = get_tablenames(ctrl)

    for ts_title in ts_titles_in_db:
        df = load_results_from_db(ctrl, ts_title)
        # all games should have been played in the same game modus
        assert len(df.gamemodus.value_counts().to_dict()) == 1
        print("Available columns are", df.columns.tolist())
        df_test = df[["player_name", "seed"]]
        assert df_test.equals(df_test.drop_duplicates(
        )), f"If we fix the salt and run a tounament twice, dublicates occure in the database. Please erase the table before rerun. This is very unlikely to occur for salt == non-deterministic."
        df = df[["player_name", "score"]]
        result = df.groupby("player_name").agg(
            [np.mean, err, np.size]).sort_values(('score', 'mean'))
        result["score", "mean+err"] = result["score",
                                             "mean"] + result["score", "err"]
        result["score", "mean-err"] = result["score",
                                             "mean"] - result["score", "err"]
        result["score", "better_than_next"] = result["score",
                                                     "mean-err"].iloc[:-1].le(result["score", "mean-err"].iloc[1:])

        print("*"*30)
        print("Note: Winner is the player with the best mean if the error bars are non-overlapping and size>~20.")
        print("      For the dull SHAIs, err==0, because they always loose the game.")
        print("-"*30)
        print(f"Scores of tournament section '{ts_title}' are:")
        print(result)
        print()


if __name__ == "__main__":

    infoTXT = "Superhirn tournament. Analysis script. Let's see who is winning here."
    parser = argparse.ArgumentParser(
        description=infoTXT, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--control_filepath", default="settings.yaml",
                        help="Control also via the yaml file. Here give location.")
    args = parser.parse_args()

    main(args.control_filepath)
