import radiomics, os, pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
from radiomics import featureextractor
from multiprocessing.pool import Pool
import argparse

join = os.path.join


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=str, default='venous')
    args = parser.parse_args()
    phase = args.phase

    phase_dict = {'arterial':'0001', 'venous':'0003'}
    # root_dir = '/data86/pancreaticCancer/LNM_Dataset/Registered2Venous/'
    # train_table = '/data86/zhengzhilin980/pancreas/PDAC_new/Patients_addN0_ct_path_info.xlsx'
    # root_dir = '/data87/pancreaticCancer/luhong_guangdong/LHGD_dataset/'
    # train_table = '/data87/pancreaticCancer/luhong_guangdong/2021-112-tianjing-guangdong-treatment.xlsx'
    root_dir = '/data87/pancreaticCancer/Media/SJU/SJU_dataset/'
    train_table = '/data87/pancreaticCancer/Media/SJU/patients.xlsx'

    train_df = pd.read_excel(train_table)
    # train_df = train_df[['LNM_identifier', 'PatientID', 'date', 'N', 'CT report N']]
    train_df = train_df[['LNM_identifier', 'N']]
    train_df['N'] = train_df['N'].map(lambda x: int(bool(x)))
    # output_table = '/data86/zhengzhilin980/pancreas/PDAC_new/%s_radiomics_features.xlsx' % phase
    # output_table = '/data87/pancreaticCancer/luhong_guangdong/%s_radiomics_features.xlsx' % phase
    output_table = '/data87/pancreaticCancer/Media/SJU/%s_radiomics_features.xlsx' % phase
    table_identifiers = train_df['LNM_identifier'].values
    file_identifiers= [filename[:-7] for filename in os.listdir(os.path.join(root_dir, 'PDAC')) if filename.endswith('.nii.gz')]
    identifiers = list(set(table_identifiers).intersection(set(file_identifiers)))
    train_df = train_df[train_df['LNM_identifier'].isin(identifiers)].reset_index(drop=True)
    # splits_file = '/data86/pancreaticCancer/nnUNETFrame/nnUNET_DATASET/nnUNET_preprocessed/Task501_LNMDataset_PDAC_classification/splits_final.pkl'
    # splits_file = '/data86/pancreaticCancer/nnUNETFrame/nnUNET_DATASET/nnUNET_preprocessed/Task504_LNMDataset_PDAC_seg_263/splits_final.pkl'

    if not os.path.exists(output_table):
        results = pd.DataFrame()

        # correctMaskSetting ={}
        correctMaskSetting = {'geometryTolerance':0.001, "correctMask": True}
        extractor = featureextractor.RadiomicsFeatureExtractor(**correctMaskSetting)
        extractor.enableAllFeatures()
        extractor.enableAllImageTypes()
        extractor.addProvenance(False)

        flists = train_df.T

        def get_featureVector(entry):
            featureVector = flists[entry]
            identifier = flists[entry]['LNM_identifier']
            print(identifier)
            image_path = join(root_dir, identifier + '_%s.nii.gz' % phase_dict[phase])
            pdac_path = join(root_dir, 'PDAC', identifier + '.nii.gz')
            # pdac_path = join(root_dir, 'PDAC', identifier + '_0003.nii.gz')
            # if not os.path.exists(pdac_path):
            #     pdac_path = join(root_dir, 'Task406_PDAC', identifier + '_0003.nii.gz')
            try:
                result = pd.Series(extractor.execute(image_path, pdac_path))
                featureVector = featureVector.append(result)
                featureVector.name = entry
            except:
                print('Error extract radiomics features')

            return featureVector


        pool = Pool()
        pool_results = pool.map(get_featureVector, list(range(len(flists.columns))))
        # pool_results = pool.map(get_featureVector, list(range(3)))

        for featureVector in pool_results:
            try:
                results = results.join(featureVector, how='outer')
            except:
                print(featureVector)

        results.T.to_excel(output_table, index=False, na_rep='NaN')

    else:
        radiomics_features_df = pd.read_excel(output_table)
        with open(splits_file, 'rb') as f:
            splits = pickle.load(f)
        all_data_path = '/data86/zhengzhilin980/pancreas/PDAC_new/Patients_for_OS.xlsx'
        # identifiers = np.concatenate([splits[0]['train'], splits[0]['val'], splits[0]['test']])
        # data_df = radiomics_features_df[radiomics_features_df['LNM_identifier'].isin(identifiers)]
        # data_df = data_df[['LNM_identifier', 'PatientID', 'date', 'N', 'CT report N']]
        # data_df.to_excel(all_data_path, index=False, na_rep='NaN')
        for i in range(len(splits)):
            splits_output_table = '/data86/zhengzhilin980/pancreas/PDAC_new/%s_radiomics_features_split%d.xlsx' % (phase, i)
            if not os.path.exists(splits_output_table):
                writer = pd.ExcelWriter(splits_output_table)
                train_identifiers = np.concatenate([splits[i]['train'], splits[i]['val']])
                test_identifiers = splits[i]['test']

                train_splits = radiomics_features_df[radiomics_features_df['LNM_identifier'].isin(train_identifiers)]
                train_splits.to_excel(writer, index=False, sheet_name='train')
                # val_splits = radiomics_features_df[radiomics_features_df['LNM_identifier'].isin(val_identifiers)]
                # val_splits.to_excel(writer, index=False, sheet_name='val')
                test_splits = radiomics_features_df[radiomics_features_df['LNM_identifier'].isin(test_identifiers)]
                test_splits.to_excel(writer, index=False, sheet_name='test')
                writer.close()















