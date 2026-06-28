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


# Full baseline set: the exact 20 train + 5 valid games the ResNet baseline used
# (getListGames('train')[:20] + getListGames('valid')[:5]). Hardcoded so this module
# imports without a SoccerNet dependency; verify_full_games() asserts it still matches.
# Used for the fair 25-game V-JEPA-vs-ResNet comparison.
VJEPA_GAMES_FULL: list[tuple[str, str]] = [
    # ── train (20) ──
    ("england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",            "train"),
    ("england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal",     "train"),
    ("england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United",  "train"),
    ("england_epl/2014-2015/2015-02-22 - 19-15 Southampton 0 - 2 Liverpool",      "train"),
    ("england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea",            "train"),
    ("england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace",     "train"),
    ("england_epl/2015-2016/2015-08-29 - 17-00 Manchester City 2 - 0 Watford",    "train"),
    ("england_epl/2015-2016/2015-09-12 - 14-45 Everton 3 - 1 Chelsea",            "train"),
    ("england_epl/2015-2016/2015-09-12 - 17-00 Crystal Palace 0 - 1 Manchester City", "train"),
    ("england_epl/2015-2016/2015-09-19 - 19-30 Manchester City 1 - 2 West Ham",   "train"),
    ("england_epl/2015-2016/2015-09-26 - 17-00 Liverpool 3 - 2 Aston Villa",      "train"),
    ("england_epl/2015-2016/2015-10-17 - 17-00 Chelsea 2 - 0 Aston Villa",        "train"),
    ("england_epl/2015-2016/2015-10-31 - 15-45 Chelsea 1 - 3 Liverpool",          "train"),
    ("england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom","train"),
    ("england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool",  "train"),
    ("england_epl/2015-2016/2015-11-29 - 15-00 Tottenham 0 - 0 Chelsea",          "train"),
    ("england_epl/2015-2016/2015-12-05 - 20-30 Chelsea 0 - 1 Bournemouth",        "train"),
    ("england_epl/2015-2016/2015-12-19 - 18-00 Chelsea 3 - 1 Sunderland",         "train"),
    ("england_epl/2015-2016/2015-12-26 - 18-00 Manchester City 4 - 1 Sunderland", "train"),
    ("england_epl/2015-2016/2016-01-03 - 16-30 Crystal Palace 0 - 3 Chelsea",     "train"),
    # ── valid (5) ──
    ("england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal",            "valid"),
    ("england_epl/2015-2016/2015-08-30 - 18-00 Swansea 2 - 1 Manchester United",  "valid"),
    ("england_epl/2015-2016/2015-09-26 - 17-00 Leicester 2 - 5 Arsenal",          "valid"),
    ("england_epl/2015-2016/2015-09-26 - 17-00 Manchester United 3 - 0 Sunderland","valid"),
    ("england_epl/2015-2016/2015-10-03 - 17-00 Manchester City 6 - 1 Newcastle Utd", "valid"),
]

VJEPA_GAME_PATHS_FULL: list[str] = [g for g, _ in VJEPA_GAMES_FULL]


def verify_full_games() -> None:
    """
    Assert VJEPA_GAMES_FULL exactly matches the baseline selection from SoccerNet.
    Raises AssertionError on drift. Requires the SoccerNet package (call when checking).
    """
    from SoccerNet.utils import getListGames

    tr = [g.replace("\\", "/") for g in getListGames("train", task="spotting")[:20]]
    va = [g.replace("\\", "/") for g in getListGames("valid", task="spotting")[:5]]
    expected = [(g, "train") for g in tr] + [(g, "valid") for g in va]
    assert VJEPA_GAMES_FULL == expected, (
        "VJEPA_GAMES_FULL drifted from baseline getListGames selection"
    )


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

    # Auto-detect under /kaggle/input. Kaggle collapses single-child directory
    # chains on dataset save, so the england_epl/<season>/ prefix may be missing
    # (e.g. .../soccernet-25games/2015-2016/<game>/...). We therefore key on the
    # unique GAME LEAF NAME only, and return the directory that contains the
    # <season>/<game> (or <game>) subtree. Use resolve_game_dir() to locate a
    # specific game inside whatever structure exists.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        leaves = {Path(g).name for g in (VJEPA_GAME_PATHS or VJEPA_GAME_PATHS_FULL)}
        for fname in ("1_224p.mkv", "1_VJEPA21_L.npy", "Labels-v2.json"):
            for found in kaggle_input.rglob(fname):
                game_leaf = found.parent.name           # the <game> folder name
                if game_leaf in leaves:
                    season = found.parent.parent.name   # e.g. 2015-2016
                    # data_dir = path up to and including england_epl, if present,
                    # else the parent of the season folder.
                    season_parent = found.parent.parent.parent
                    return str(season_parent)
    return "./data/soccernet"


def resolve_game_dir(data_dir: str, game: str) -> Path:
    """
    Find the actual directory for a game under data_dir, tolerant of Kaggle's
    folder flattening. `game` is the canonical 'england_epl/<season>/<name>' path.

    Tries, in order:
      1. data_dir/england_epl/<season>/<name>   (canonical)
      2. data_dir/<season>/<name>               (england_epl collapsed)
      3. data_dir/<name>                         (season collapsed too)
      4. recursive search for the unique <name> leaf folder.
    Returns the first existing path; falls back to the canonical join.
    """
    root = Path(data_dir)
    parts = Path(game).parts          # (england_epl, <season>, <name>)
    name = parts[-1]
    season = parts[-2] if len(parts) >= 2 else None

    candidates = [root / game]
    if season:
        candidates.append(root / season / name)
    candidates.append(root / name)
    for c in candidates:
        if c.exists():
            return c

    # last resort: find the unique leaf folder anywhere under root
    for d in root.rglob(name):
        if d.is_dir():
            return d
    return root / game                # canonical fallback (will error informatively downstream)


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
