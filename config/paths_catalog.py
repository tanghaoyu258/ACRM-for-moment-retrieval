import os

from .settings import (
    ANET_FEATURES_PATH,
    CHARADES_FEATURES_PATH,
    EMBEDDINGS_PATH,
    ANNOTATIONS_PATH)


class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "anet_cap_train": {
            "feature_path": os.path.join(
                ANET_FEATURES_PATH, 'anet-cap/anet_c3d.hdf5'),
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'anet-cap/train.json'),
            "embeddings_path": os.path.join(
                EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
        },

        "anet_cap_test": {
            "feature_path": os.path.join(
                ANET_FEATURES_PATH, 'anet-cap/anet_c3d.hdf5'),
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'anet-cap/val_2.json'),
            "embeddings_path":
                os.path.join(
                    EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
        },

        "charades_sta_train": {
            "feature_path": os.path.join(
                CHARADES_FEATURES_PATH, 'charades-sta/charades_i3d.hdf5'),
            "ann_file_path":
                os.path.join(
                    ANNOTATIONS_PATH, 'charades-sta/charades_sta_train_tokens.json'),
            "embeddings_path": os.path.join(
                EMBEDDINGS_PATH, 'glove.840B.300d.txt')
        },

        "charades_sta_test": {
            "feature_path": os.path.join(
                CHARADES_FEATURES_PATH, 'charades-sta/charades_i3d.hdf5'),
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'charades-sta/charades_sta_test_tokens.json'),
            "embeddings_path": os.path.join(
                EMBEDDINGS_PATH, 'glove.840B.300d.txt')
        },
        "tacos_train": {
            "feature_path": os.path.join(
                ANET_FEATURES_PATH, 'tacos/tacos_c3d.hdf5'),
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'tacos/tacos_train.json'),
            "embeddings_path": os.path.join(
                EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
        },

        "tacos_test": {
            "feature_path": os.path.join(
                ANET_FEATURES_PATH, 'tacos/tacos_c3d.hdf5'),
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'tacos/tacos_test.json'),
            "embeddings_path":
                os.path.join(
                    EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
        },
    }

    @staticmethod
    def get(name):
        if "charades_sta" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                # features_path=os.path.join(data_dir, attrs["feature_path"]),
                features_path=attrs["feature_path"],
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(data_dir, attrs["embeddings_path"]),
            )
            return dict(
                factory="CHARADES_STA",
                args=args,
            )
        if "anet_cap" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                # features_path=os.path.join(data_dir, attrs["feature_path"]),
                features_path=attrs["feature_path"],
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(data_dir, attrs["embeddings_path"]),
            )
            return dict(
                factory="ANET_CAP",
                args=args,
            )
        if "tacos" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                # features_path=os.path.join(data_dir, attrs["feature_path"]),
                features_path=attrs["feature_path"],
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(data_dir, attrs["embeddings_path"]),
            )
            return dict(
                factory="TACOS",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
