import os
from multiprocessing.pool import ThreadPool
import SimpleITK as sitk
import numpy as np
import collections
import json
import matplotlib.pyplot as plt
import argparse
join = os.path.join


class DistanceTransform(object):
    def __init__(self, label_dir, save_dir):
        self.organ_vessel_label_dict = {'spleen': 1, 'right_kidney': 2, 'left_kidney': 3, 'gallbladder': 4, 'esophagus': 5, 'liver': 6, 'stomach': 7, 'aorta': 8, \
                             'inferior_vena_cava': 9, 'portal_vein_splenic_vein': 10, 'pancreas': 11, 'right_adrenal_gland': 12, 'left_adrenal_gland': 13, 'duodenum': 14, \
                             'superior_mesenteric_vein': 15, 'superior_mesenteric_artery': 16, 'truncus_coeliacus': 17, 'left_gastric_artery': 18, 'PDAC': 19, 'proper_hepatic_artery_common_hepatic_artery': 20}
        self.LN_18_group_dict = {'esophagus': [1,2], 'stomach':[1,2,3,4,5,6], 'duodenum':[5,6, 12, 13,17], 'left_gastric_artery':[7], 'proper_hepatic_artery_common_hepatic_artery':[8],
                                 'truncus_coeliacus':[9, 11], 'spleen':[10], 'superior_mesenteric_artery':[14], 'aorta':[16], 'pancreas':[18]}
        self.label_dir = label_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.filter = self.get_filter()
    def get_filter(self):
        filter = sitk.SignedMaurerDistanceMapImageFilter()
        filter.SetUseImageSpacing(True)
        filter.SetSquaredDistance(False)
        return filter

    def get_single_distance_transform(self, label_sitk, save_path=None):
        if save_path and not os.path.exists(save_path):
            dmap = self.filter.Execute(label_sitk)
            sitk.WriteImage(dmap, save_path)
        elif save_path is None:
            return self.filter.Execute(label_sitk)

    def get_multilabel_distance_transform(self, case_identifier):
        label_sitk = sitk.ReadImage(join(self.label_dir, case_identifier + '.nii.gz'))
        for label_name in self.LN_18_group_dict.keys():
            print("----- Computing distance transform for {} {} ----".format(case_identifier, label_name))
            save_path = join(self.save_dir, case_identifier + '_%s.nii.gz' % label_name)
            if not os.path.exists(save_path):
                label_id = self.organ_vessel_label_dict[label_name]
                if label_name != 'pancreas':
                    label_id_np = (sitk.GetArrayFromImage(label_sitk) == label_id).astype(int)
                else: # merge pancreas and PDAC
                    label_id_np = ((sitk.GetArrayFromImage(label_sitk) == label_id)|(sitk.GetArrayFromImage(label_sitk) == 19)).astype(int)
                if label_id_np.any():
                    label_id_sitk = sitk.GetImageFromArray(label_id_np)
                    label_id_sitk.CopyInformation(label_sitk)
                    self.get_single_distance_transform(label_id_sitk, save_path)

    def get_distance_transform_all(self, num_parts=1, part_id=0):
        case_identifiers = [filename[:-7] for filename in os.listdir(self.label_dir) if filename.endswith('.nii.gz')]
        # for case_identifier in case_identifiers[part_id::num_parts]:
        #     self.get_multilabel_distance_transform(case_identifier)
        p = ThreadPool()
        p.map(self.get_multilabel_distance_transform, case_identifiers[part_id::num_parts])
        p.close()
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_dir", help="directory of organ&vessel masks")
    parser.add_argument("dmap_dir", help="directory of output distance maps")
    parser.add_argument("--num_parts", type=int, required=False, default=1)
    parser.add_argument("--part_id", type=int, required=False, default=0)
    args = parser.parse_args()
    mask_dir = args.mask_dir
    dmap_dir = args.dmap_dir
    num_parts = args.num_parts
    part_id = args.part_id

    if not os.path.exists(dmap_dir):
        os.makedirs(dmap_dir)

    DT = DistanceTransform(label_dir=mask_dir, save_dir=dmap_dir)
    DT.get_distance_transform_all(num_parts=num_parts, part_id=part_id)
