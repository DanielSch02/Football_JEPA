"""
Download SoccerNet 224p match videos (+ labels) for the small V-JEPA experiment.

Unlike scripts/load_data.py (which fetches precomputed ResNET_TF2 features for the
full baseline split), this fetches the raw 1_224p.mkv / 2_224p.mkv videos that
V-JEPA 2.1 needs as pixel input, for just the 3 games in src/config.VJEPA_GAMES.

The NDA password is read from Kaggle Secrets when available (see get_nda_password),
so this runs unattended on Kaggle. Already-downloaded files are skipped, making the
script resumable.

Run from the project root:
    python -m scripts.load_videos
On Kaggle the videos download to /kaggle/working/soccernet (set DATA_DIR / env).
"""

import sys
from pathlib import Path

# Suppress GA telemetry ping that times out on some networks (same as load_data.py)
import google_measurement_protocol
google_measurement_protocol.report = lambda *a, **kw: []

from SoccerNet.Downloader import SoccerNetDownloader

from footy.config import VJEPA_GAMES, VJEPA_GAMES_FULL, get_nda_password

# Default local target; on Kaggle pass --data_dir /kaggle/working/soccernet
DATA_DIR = "./data/soccernet"
VIDEO_FILES = ["Labels-v2.json", "1_224p.mkv", "2_224p.mkv"]


def download(data_dir: str = DATA_DIR, games: list[tuple[str, str]] = VJEPA_GAMES) -> None:
    d = SoccerNetDownloader(data_dir)
    d.password = get_nda_password()

    print(f"Downloading {len(games)} games to {data_dir}")
    for i, (game, split) in enumerate(games, 1):
        game_path = Path(data_dir) / game
        print(f"[{i}/{len(games)}] [{split}] {game}")

        for fname in VIDEO_FILES:
            if (game_path / fname).exists():
                print(f"    {fname} already present, skipping")
                continue
            d.downloadGame(game=game, files=[fname], spl=split)
            print(f"    downloaded {fname}")

    print("\nDone. Summary:")
    for game, _ in games:
        gp = Path(data_dir) / game
        have = [f for f in VIDEO_FILES if (gp / f).exists()]
        print(f"  {len(have)}/{len(VIDEO_FILES)}  {game}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--full", action="store_true",
                        help="Use the full 25-game baseline set (default: 3-game smoke set)")
    args = parser.parse_args()
    download(args.data_dir, VJEPA_GAMES_FULL if args.full else VJEPA_GAMES)
