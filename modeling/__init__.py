from modeling.localization import Localization
from modeling.localization_ACRM import Localization_ACRM
import torch
def build(cfg):
    if cfg.MODEL_NAME == 'TMLGA':
        return Localization(cfg)
    elif cfg.MODEL_NAME == 'ACRM':
        return Localization_ACRM(cfg)


