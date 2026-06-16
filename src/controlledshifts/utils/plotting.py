"""Shared matplotlib/seaborn styling helpers.

Centralizes the figure styling used across the analysis runners and configures the global font so every figure
renders in DM Sans when it is installed on the system. DM Sans is not shipped with the project; if it is not found,
figures fall back to matplotlib's default sans-serif font and a single warning is logged.
"""

from logging import Logger

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from controlledshifts.utils.pylogger import get_pylogger


_log = get_pylogger(__name__)

# Preferred figure font, followed by the fallbacks matplotlib should try when it (or a glyph) is unavailable.
DEFAULT_FONT = "DM Sans"
FALLBACK_FONTS = ["DejaVu Sans", "sans-serif"]


def configure_fonts(font: str = DEFAULT_FONT, log: Logger | None = None) -> bool:
    """Set the global matplotlib font to ``font`` if it is installed, otherwise leave the defaults untouched.

    ``rcParams`` is process-global, so a single call applies to every figure created afterwards in the same process.
    When the font is missing the function does not modify ``rcParams`` and logs one warning, so figures degrade
    gracefully to matplotlib's default sans-serif font instead of crashing.

    Args:
        font: Family name to look up and apply (as reported by the OS / fontconfig).
        log: Logger for the "font not found" warning; falls back to this module's logger when omitted.

    Returns:
        ``True`` if ``font`` was found and applied, ``False`` otherwise.
    """
    log = log or _log
    available = {f.name for f in font_manager.fontManager.ttflist}
    if font not in available:
        log.warning("Font '%s' not found; figures will use the default font.", font)
        return False

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font, *FALLBACK_FONTS]
    return True


def set_analysis_theme(log: Logger | None = None) -> None:
    """Apply the shared analysis figure theme (seaborn whitegrid, talk context) and the DM Sans font.

    This is the single source of truth for the styling previously duplicated across the analysis runners. The font is
    configured last because ``sns.set_theme`` resets ``font.family``/``font.sans-serif`` to seaborn's defaults.

    Args:
        log: Logger forwarded to :func:`configure_fonts` for the "font not found" warning.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        },
    )
    configure_fonts(log=log)
