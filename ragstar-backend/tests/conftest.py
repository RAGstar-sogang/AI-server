from pathlib import Path
import sys


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))


from set_model_of_live_tests import *