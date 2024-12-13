import os

CACHE_DIR=os.environ.get('CACHE_DIR', None)
CODE_DIR=os.environ.get('CODE_DIR', None)
IMAGE_DATA_DIR=os.environ.get('IMAGE_DATA_DIR', None)
IMAGE_INSTRUCTION_DIR=os.environ.get('IMAGE_INSTRUCTION_DIR', None)
DATA_DIR = os.environ.get('DATA_DIR', None)

paths={
    'CACHE_DIR': CACHE_DIR,
    'CODE_DIR': CODE_DIR,
    'IMAGE_DATA_DIR': IMAGE_DATA_DIR,
    'IMAGE_INSTRUCTION_DIR': IMAGE_INSTRUCTION_DIR,
    'DATA_DIR': DATA_DIR
}

# dataset_URLs = {
#     'MMStar': 'https://huggingface.co/datasets/Lin-Chen/MMStar/resolve/main/MMStar.tsv'
# }

# img_root_map = {k: k for k in dataset_URLs}
# img_root_map.update({
#     'MMStar': 'MMStar'
# })

# assert set(dataset_URLs) == set(img_root_map)

# def LMUDataRoot():
#     IMAGE_DATA_DIR=os.environ.get('IMAGE_DATA_DIR', None)
#     assert IMAGE_DATA_DIR is not None
#     root = osp.join(IMAGE_DATA_DIR, 'LMU')
#     os.makedirs(root, exist_ok=True)
#     return root

# # reuse 
# from vlmeval.smp.misc import *
# from vlmeval.smp.file import *
# from vlmeval.utils import DATASET_TYPE
# our define
from .image_utils import *
from .utils import *
