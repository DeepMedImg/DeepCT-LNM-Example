import os
import numpy as np
from multiprocessing.pool import ThreadPool
import argparse
# import matplotlib.pyplot as plt
from functools import partial
import SimpleITK as sitk

join = os.path.join


def nonlinear_function(x, minimum, maximum):
    if x <= maximum and x >= minimum:
        return 1.
    elif x > maximum and x <= maximum + 5:
        return -1. / 5. * (x - maximum - 5)
    elif x < minimum and x >= minimum - 5:
        return 1. / 5. * (x - minimum + 5)
    else:
        return 0.


def get_attmap(dmap_files, save=True):
    '''
    Get attention map according to distance transform of organs and vessels
    '''

    distance_dict = {'esophagus': (0, 25), 'stomach': (-2, 18), 'duodenum': (-5, 22),
                                 'left_gastric_artery': (0, 21), 'proper_hepatic_artery_common_hepatic_artery': (0,20),
                                 'truncus_coeliacus': (-2, 18), 'spleen': (0,16), 'superior_mesenteric_artery': (-1, 20),
                                 'aorta': (0, 28), 'pancreas': (-5, 20)}
    if not os.path.exists(join(attmap_dir, '_'.join(dmap_files[0].split('_')[:2]) + '_attmap.nii.gz')):
    # if not os.path.exists(join(attmap_dir, '_'.join(dmap_files[0].split('_')[:3]) + '_attmap.nii.gz')):
        attmaps = []
        dmap_sitk0 = sitk.ReadImage(join(dmap_dir, dmap_files[0]))
        shape = sitk.GetArrayFromImage(dmap_sitk0).shape
        refined_attmap = np.full(shape=shape, fill_value=-1.)
        for dmap_file in dmap_files:
            # vessel = dmap_file[10:-7]
            # vessel = dmap_file[17:-7]
            # vessel = dmap_file[5:-7]
            # dmap_identifier = '_'.join(dmap_file.split('_')[:3])
            dmap_identifier = '_'.join(dmap_file.split('_')[:2])
            vessel = dmap_file[len(dmap_identifier)+1:-7]
            minimum, maximum = distance_dict[vessel]
            dmap_path = join(dmap_dir, dmap_file)
            dmap_sitk = sitk.ReadImage(dmap_path)
            dmap_np = sitk.GetArrayFromImage(dmap_sitk)

            # nonlinear = partial(nonlinear_function, minimum=minimum, maximum=maximum)
            # v_nonlinear = np.vectorize(nonlinear)
            # attmap1 = v_nonlinear(dmap_np)
            # use np.piecewise instead of np.vectorize
            attmap2 = np.piecewise(dmap_np, [(dmap_np <= maximum)&(dmap_np >= minimum), (dmap_np > maximum)&(dmap_np <= maximum + 3), (dmap_np < minimum)&(dmap_np >= minimum - 3)],
                                  [1., lambda x: -1. / 3. * (x - maximum - 3), lambda x:1. / 3. * (x - minimum + 3), 0.])
            refined_attmap[dmap_np < 0] = attmap2[dmap_np < 0]
            attmaps.append(attmap2)
        attmaps = np.stack(attmaps, axis=0)
        refined_attmap[refined_attmap == -1] = np.max(attmaps, axis=0)[refined_attmap == -1]
        # print('_'.join(dmap_files[0].split('_')[:2]))
        print(dmap_identifier)
        if save:
            # for idx, dmap_file in enumerate(dmap_files):
            #     vessel = dmap_file[5:-7]
            #     attmap = attmaps[idx]
            #     attmap_sitk = sitk.GetImageFromArray(attmap.astype(np.float32))
            #     attmap_sitk.CopyInformation(dmap_sitk0)
            #     sitk.WriteImage(attmap_sitk,join(attmap_dir, '_'.join(dmap_files[0].split('_')[:1]) + '_%s_attmap.nii.gz' % vessel))
            refined_attmap_sitk = sitk.GetImageFromArray(refined_attmap.astype(np.float32))
            refined_attmap_sitk.CopyInformation(dmap_sitk0)
            sitk.WriteImage(refined_attmap_sitk, join(attmap_dir, '_'.join(dmap_files[0].split('_')[:2]) + '_attmap.nii.gz'))
            # sitk.WriteImage(refined_attmap_sitk, join(attmap_dir, '_'.join(dmap_files[0].split('_')[:1]) + '_attmap.nii.gz'))
            # sitk.WriteImage(refined_attmap_sitk, join(attmap_dir, '_'.join(dmap_files[0].split('_')[:3]) + '_attmap.nii.gz'))
        return refined_attmap


def get_attmap_all(image_dir, dmap_dir, num_parts=1, part_id=0):
    case_identifiers = np.unique([filename[:-12] for filename in os.listdir(image_dir) if filename.endswith('.nii.gz')])
    # case_identifiers = ['pdac_0017']
    dmap_files_list = []
    for case_identifier in case_identifiers:
        dmap_files_list.append([filename for filename in os.listdir(dmap_dir) if filename.startswith(case_identifier) and filename.endswith('nii.gz')])
    p = ThreadPool(16)
    p.map(get_attmap, dmap_files_list[part_id::num_parts])
    p.close()
    p.join()
    # for dmap_files in dmap_files_list[part_id::num_parts]:
    #     get_attmap(dmap_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="directory of images")
    parser.add_argument("dmap_dir", help="directory of distance maps")
    parser.add_argument("attmap_dir", help="directory of output attention maps")
    parser.add_argument("--num_parts", type=int, required=False, default=1)
    parser.add_argument("--part_id", type=int, required=False, default=0)
    args = parser.parse_args()
    image_dir = args.image_dir
    dmap_dir = args.dmap_dir
    attmap_dir = args.attmap_dir
    num_parts = args.num_parts
    part_id = args.part_id

    if not os.path.exists(attmap_dir):
        os.makedirs(attmap_dir)

    get_attmap_all(image_dir, dmap_dir, num_parts=num_parts, part_id=part_id)


