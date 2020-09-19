import os

CODE_ROOT = os.path.dirname(os.path.realpath(__file__))
ANNOTATIONS_PATH = os.path.join(os.path.dirname(CODE_ROOT), "preprocessing")

HOME = os.environ["HOME"]
# /disk/thy/WSDEC/TGNmodel/glove.6B.300d.txt
DATA_PATH = os.path.join(HOME, "data", "TMLGA")

ANET_FEATURES_PATH = ANNOTATIONS_PATH
CHARADES_FEATURES_PATH = ANNOTATIONS_PATH
EMBEDDINGS_PATH = os.path.join(DATA_PATH, "word_embeddings")
# EMBEDDINGS_PATH = '/disk/thy/WSDEC/TGNmodel/'
