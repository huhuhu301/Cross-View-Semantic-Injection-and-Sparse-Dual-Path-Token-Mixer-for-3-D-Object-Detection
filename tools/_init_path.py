# Modified by RV-SDTM contributors for this public release; see NOTICE.
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)
