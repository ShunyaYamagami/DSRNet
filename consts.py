import os

DATASET_PARENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets_okamoto"))
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "datasets"))
LABEL_STUDIO_ROOT = os.path.abspath(os.path.join(DATASET_ROOT, "label_studio/VID_20240124_163346"))
CHECKPOINTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
