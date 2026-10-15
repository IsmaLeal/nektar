"""Save a figure and, next to it, a .cmd file with the command line that made
it, so every figure under cases/se/figures/ can be regenerated:

    from figsave import save
    save(fig, out)                  # one file, format from the suffix
    save(fig, out, also_png=True)   # out (pdf) and out.with_suffix('.png')

out.with_suffix('.cmd') holds the working directory and the argv line."""
import os
import shlex
import sys
from pathlib import Path


def save(fig, out, dpi=200, also_png=False):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    if also_png and out.suffix != ".png":
        fig.savefig(out.with_suffix(".png"), dpi=dpi)
    out.with_suffix(".cmd").write_text(f"cd {os.getcwd()}\npython3 {shlex.join(sys.argv)}\n")
    return out
