from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class PascalPart108(CustomDataset):
    CLASSES = ('background', 'aero body', 'aero stern', 'aero rwing','aero engine', 'aero wheel',
              'bike fwheel', 'bike saddle', 'bike handlebar', 'bike chainwheel', 'birds head',
              'birds beak', 'birds torso', 'birds neck', 'birds rwing', 'birds rleg', 'birds rfoot',
              'birds tail', 'boat', 'bottle cap', 'bottle body', 'bus rightside', 'bus roofside', 
              'bus mirror', 'bus fliplate', 'bus door', 'bus wheel', 'bus headlight', 'bus window', 
              'car rightside', 'car roofside', 'car fliplate', 'car door', 'car wheel', 'car headlight', 
              'car window', 'cat head', 'cat reye', 'cat rear', 'cat nose', 'cat torso', 'cat neck', 
              'cat rfleg', 'cat rfpaw', 'cat tail', 'chair', 'cow head', 'cow rear', 'cow muzzle', 
              'cow rhorn', 'cow torso', 'cow neck', 'cow rfuleg', 'cow tail', 'dining table', 'dog head', 
              'dog reye', 'dog rear', 'dog nose', 'dog torso', 'dog neck', 'dog rfleg', 'dog rfpaw', 
              'dog tail', 'dog muzzle', 'horse head', 'horse rear', 'horse muzzle', 'horse torso', 
              'horse neck', 'horse rfuleg', 'horse tail', 'horse rfho', 'mbike fwheel', 'mbike hbar',
              'mbike saddle', 'mbike hlight', 'person head', 'person reye', 'person rear', 'person nose', 
              'person mouth', 'person hair', 'person torso', 'person neck', 'person ruarm', 'person rhand', 
              'person ruleg', 'person rfoot', 'pplant pot', 'pplant plant', 'sheep head', 'sheep rear', 
              'sheep muzzle', 'sheep rhorn', 'sheep torso', 'sheep neck', 'sheep rfuleg', 'sheep tail', 
              'sofa', 'train head', 'train hrightside', 'train hroofside', 'train headlight', 'train coach', 
              'train crightside', 'train croofside', 'tv screen')
   
    PALETTE = [[0, 0, 0], [0, 192, 64], [0, 64, 96], [128, 192, 192],
               [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
               [64, 128, 32], [0, 160, 0], [0, 192, 64], [192, 128, 160],
               [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
               [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
               [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
               [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
               [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
               [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
               [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
               [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
               [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
               [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
               [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
               [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
               [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
               [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
               [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
               [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
               [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
               [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
               [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192]]


    def __init__(self, **kwargs):
        super(PascalPart108, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False, 
            **kwargs)