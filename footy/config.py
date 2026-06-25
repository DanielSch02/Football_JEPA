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

    Resolution order:
      1. SOCCERNET_DIR env var (explicit override),
      2. the first Kaggle input mount that contains the expected game folders
         (handles arbitrary dataset nesting like
         /kaggle/input/datasets/<user>/<slug>/soccernet/england_epl/...),
      3. local ./data/soccernet.
    """
    env = os.environ.get("SOCCERNET_DIR")
    if env:
        return env

    # Auto-detect under /kaggle/input by locating a Labels-v2.json whose path ends
    # with the known game-relative path, then strip that suffix to get data_dir.
    probe = VJEPA_GAME_PATHS[0]                 # e.g. england_epl/2014-2015/<game>
    suffix = Path(probe) / "Labels-v2.json"
    n_strip = len(suffix.parts)                 # parts to drop from the matched file path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for labels in kaggle_input.rglob("Labels-v2.json"):
            if labels.parts[-n_strip:] == suffix.parts:
                return str(Path(*labels.parts[:-n_strip]))

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
