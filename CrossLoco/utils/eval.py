
import numpy as np
from scipy import linalg

FILE_NAMES =['./saved_robot_motion/baseline_traj/jumps1_subject1_180_300_',
             './saved_robot_motion/baseline_traj/run2_subject4_1000_300_',
             './saved_robot_motion/baseline_traj/walk2_subject3_200_300_',
             './saved_robot_motion/baseline_traj/walk3_subject3_2100_300_',
             './saved_robot_motion/baseline_traj/walk3_subject3_4550_300_']


def evaluate_diversity(method_name, diversity_times=500):
    files = []
    for file_name in FILE_NAMES:
        files.append(np.load(file_name+method_name+'.npy'))
    a,b,c = np.shape(files)
    activation = np.reshape(np.array(files),(int(a*b),c))
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)

    print(f'---> [{method_name}] Diversity Mean: {dist.mean():.4f}')
    print(f'---> [{method_name}] Diversity Std: {dist.std()/2:.4f}')




evaluate_diversity(method_name='crossloco')