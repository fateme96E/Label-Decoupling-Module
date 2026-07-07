from experiments.scp_experiment_decoupled import SCP_Experiment
from utils import utils
# model configs
from configs.configs import *
from pathlib import Path
from datetime import datetime
import pandas as pd
import shutil
import json


def main():
    
    datafolder = '../data/ptbxl/'
    outputfolder = '../check/'
    # outputfolder = '../outputs_CBL/'
    # archive_root = "../my_cbl_run_records/"  # separate personal folder

    models = [
        # conf_decoupled_fastai_xresnet1d101_no_metric,
        # conf_fastai_xresnet1d101_cbl,
        # conf_decoupled_fastai_xresnet1d101_cbl,
        # conf_fastai_xresnet1d101,
        # conf_fastai_resnet1d_wang,
        # conf_fastai_lstm,
        # conf_fastai_lstm_bidir,
        # conf_fastai_fcn_wang,
        # conf_fastai_inception1d,
        conf_decoupled_fastai_xresnet1d101,
        # conf_decoupled_fastai_resnet1d_wang,
        # conf_decoupled_fastai_lstm,
        # conf_decoupled_fastai_lstm_bidir,
        # conf_decoupled_fastai_fcn_wang,
        # conf_decoupled_fastai_inception1d,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
       ]
       


    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        e.perform()
        e.evaluate(output_name='val_result.csv')

    # generate greate summary table
    utils.generate_ptbxl_summary_table(folder=outputfolder)



if __name__ == "__main__":
    main()
