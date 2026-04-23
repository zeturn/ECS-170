'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting

class Setting_KFold_CV(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        raise RuntimeError('Stage 2 dataset already provides train/test split. Please use Setting_Train_Test_Split.')

        