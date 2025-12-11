import os
import numpy as np
import pickle
from scipy.spatial.transform import Rotation


keywords = ['walk','dance','run', 'sprint','fight','jump']
lafan_inputdir = '/media/bic/77223f40-ef66-482e-aa9e-ad9ea5a67660/LAFAN1_Retargeting_Dataset/g1'
mimickit_outputdir = '/media/bic/8258436c-e92f-4217-a894-82b40470b374/Grain/grain/Yangchen/Mimickit_raw/data/motions/lafan_sub'



# debug
# with open('data/motions/g1/g1_run.pkl', "rb") as filestream:
# with open('data/motions/humanoid/long/humanoid_long_walk0_mirror.pkl', "rb") as filestream:
#     in_dict = pickle.load(filestream)
#     print(in_dict.keys())
#     print(len(in_dict['frames']))
#     print(len(in_dict['frames'][0]))
#     print(in_dict['loop_mode'])
#     print(in_dict['fps'])

# humanoid_files = [f for f in os.listdir('data/motions/humanoid/long') if f.endswith('.pkl')]
# for file_name in humanoid_files:
#     with open(os.path.join('data/motions/humanoid/long', file_name), "rb") as filestream:
#         in_dict = pickle.load(filestream)
#         print(file_name, len(in_dict['frames'][0]), in_dict['loop_mode'], in_dict['fps'], type(in_dict['frames'][0][0]))

if not os.path.exists(mimickit_outputdir):
        print('no output dir')
        exit()
if not os.path.exists(lafan_inputdir):
        print('no input dir')
        exit()

lafan_csv_files = [f for f in os.listdir(lafan_inputdir) if f.endswith('.csv')]
for file_name in lafan_csv_files:
    if len(keywords) > 0:
        is_match = False
        for kw in keywords:
            if kw in file_name:
                is_match = True
                break
        if not is_match:
            continue

    src_path = os.path.join(lafan_inputdir, file_name)
    dest_filename = file_name.replace('.csv', '.pkl')
    dest_path = os.path.join(mimickit_outputdir, dest_filename)

    data = np.loadtxt(src_path, delimiter=',')
    root_t = data[:, 0:3]
    root_q = data[:, 3:7]
    pose = data[:, 7:]

    root_r_rotvec = Rotation.from_quat(root_q).as_rotvec()
    mimickit_data = np.concatenate([root_t, root_r_rotvec, pose], axis=-1)

    out_dict = {'loop_mode': 0, 'fps': 120, 'frames': mimickit_data}

    with open(dest_path, "wb") as f:
        pickle.dump(out_dict, f)
    
    print(f'converted {dest_path}')
