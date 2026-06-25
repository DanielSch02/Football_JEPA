"""
Downloads Labels-v2.json and ResNET_TF2.npy features for a set of games.
Run from the project root:
    python -m scripts.load_data
"""

import sys
import unittest.mock

# Suppress GA telemetry ping that times out on some networks
import google_measurement_protocol
google_measurement_protocol.report = lambda *a, **kw: []

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from pathlib import Path

DATA_DIR = "./data/soccernet"
N_TRAIN_GAMES = 20
N_VALID_GAMES = 5   # small validation set for evaluation

d = SoccerNetDownloader(DATA_DIR)
d.password = input("Enter your SoccerNet NDA password: ")

train_games = getListGames("train", task="spotting")[:N_TRAIN_GAMES]
valid_games = getListGames("valid", task="spotting")[:N_VALID_GAMES]

print(f"\nDownloading {N_TRAIN_GAMES} train games + {N_VALID_GAMES} valid games")
print(f"Estimated size: ~{(N_TRAIN_GAMES + N_VALID_GAMES) * 2 * 44:.0f} MB\n")

for split, games in [("train", train_games), ("valid", valid_games)]:
    for i, game in enumerate(games, 1):
        game_path = Path(DATA_DIR) / game
        label_done = (game_path / "Labels-v2.json").exists()
        feat1_done = (game_path / "1_ResNET_TF2.npy").exists()
        feat2_done = (game_path / "2_ResNET_TF2.npy").exists()

        print(f"[{split} {i}/{len(games)}] {game}")

        if not label_done:
            d.downloadGame(game=game, files=["Labels-v2.json"], spl=split)
        else:
            print("  Labels-v2.json already exists, skipping")

        if not feat1_done:
            d.downloadGame(game=game, files=["1_ResNET_TF2.npy"], spl=split)
        else:
            print("  1_ResNET_TF2.npy already exists, skipping")

        if not feat2_done:
            d.downloadGame(game=game, files=["2_ResNET_TF2.npy"], spl=split)
        else:
            print("  2_ResNET_TF2.npy already exists, skipping")

print("\nDone. Summary:")
for split, games in [("train", train_games), ("valid", valid_games)]:
    complete = sum(
        1 for g in games
        if (Path(DATA_DIR) / g / "Labels-v2.json").exists()
        and (Path(DATA_DIR) / g / "1_ResNET_TF2.npy").exists()
        and (Path(DATA_DIR) / g / "2_ResNET_TF2.npy").exists()
    )
    print(f"  {split}: {complete}/{len(games)} games fully downloaded")
