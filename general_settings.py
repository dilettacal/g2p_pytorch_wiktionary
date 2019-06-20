from root_path import get_full_path

DATA_DIR = get_full_path("data")
DATA_DIR_PREPRO = get_full_path("data", "preprocessed")
WIKI_DIR = get_full_path("data", "wiki")

import os
import stat
import sys


def check_permissions(start_dir):
    for root, dirs, files in os.walk(start_dir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o777)


def permission_checker(dir):
    if sys.platform == "Windows" or sys.platform.lower().startswith("win"):
        os.chmod(get_full_path(dir), stat.S_IWRITE)  ### Windows problems on directory rights
    else:
        check_permissions(".")
