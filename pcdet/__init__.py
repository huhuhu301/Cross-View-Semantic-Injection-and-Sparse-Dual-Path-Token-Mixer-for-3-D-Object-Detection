# Modified by RV-SDTM contributors for this public release; see NOTICE.
import subprocess
from pathlib import Path

from .version import __version__

__all__ = [
    '__version__'
]


def get_git_commit_number():
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / '.git').exists():
        return '0000000'

    cmd_out = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=str(repo_root),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        check=False,
    )
    if cmd_out.returncode != 0:
        return '0000000'
    return cmd_out.stdout.strip()[:7] or '0000000'


script_version = get_git_commit_number()


if script_version != '0000000' and script_version not in __version__:
    __version__ = __version__ + '+py%s' % script_version
