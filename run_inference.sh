#! /bin/bash
python nnunet/experiment_planning/nnUNet_plan_and_preprocess_pdac_classification.py $1 $1/cropped $1/preprocessed 0 -label_table $1/Patients.xlsx -ln_mask_dir $1/LN_stage2/fold_0 -pl2d ExperimentPlanner2D_v21_pdac_classification &&\
python nnunet/inference/predict_simple_lnm_pred.py 2d nnUNetTrainerV2_PDAC_classification_resnet_MaskSideBranch_SegLN_v2 501 0 $1/preprocessed --valbest_ma --val_folder testing_raw_MA

