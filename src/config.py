"""
Shared configuration for the V-JEPA feature path.

Single source of truth for:
  - which games make up the small V-JEPA experiment (game path + SoccerNet split),
  - the data directory resolution (local vs Kaggle),
  - the NDA password lookup (Kaggle Secrets, falling back to an interactive prompt).

The 3 games below all have ResNet features + labels already downloaded locally, so
V-JEPA features extracted for them can be compared apples-to-apples against the
existing ResNet baseline. Splits are verified against SoccerNet.getListGames():
Burnley 0-1 Arsenal is in the *valid* split, the other two are in *train*.
"""

import os
from pathlib import Path

# (game_path, soccernet_split) — split matters for the SoccerNet downloader.
VJEPA_GAMES: list[tuple[str, str]] = [
    ("england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",        "train"),
    ("england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal", "train"),
    ("england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal",        "valid"),
]

# Game paths only (no split) — for code that just iterates games.
VJEPA_GAME_PATHS: list[str] = [g for g, _ in VJEPA_GAMES]


def default_data_dir() -> str:
    """
    Resolve the SoccerNet data directory.

    On Kaggle the attached '3-matches' dataset is read-only at
    /kaggle/input/3-matches; we set SOCCERNET_DIR there in the notebook.
    Locally it falls back to ./data/soccernet.
    """
    env = os.environ.get("SOCCERNET_DIR")
    if env:
        return env
    kaggle_input = Path("/kaggle/input/3-matches")
    if kaggle_input.exists():
        return str(kaggle_input)
    return "./data/soccernet"


def get_nda_password() -> str:
    """
    Return the SoccerNet NDA password.

    Order of resolution:
      1. Kaggle Secret named SOCCERNET_PWD (when running on Kaggle),
      2. SOCCERNET_PWD environment variable,
      3. interactive input() prompt (local use).
    """
    try:
        from kaggle_secrets import UserSecretsClient

        pwd = UserSecretsClient().get_secret("SOCCERNET_PWD")
        if pwd:
            return pwd
    except Exception:
        pass  # not on Kaggle, or secret not attached — fall through

    env = os.environ.get("SOCCERNET_PWD")
    if env:
        return env

    return input("Enter your SoccerNet NDA password: ")
