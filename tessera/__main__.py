"""Start the Tessera desktop app: ``python -m tessera [proje.tessera.json | görsel]``."""
import sys

from tessera.ui import run

if __name__ == "__main__":
    sys.exit(run(sys.argv))
