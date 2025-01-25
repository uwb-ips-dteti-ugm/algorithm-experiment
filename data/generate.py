import os
import numpy as np
import yaml
import csv
import uuid

GENERATE_YAML       = yaml.safe_load(open('./generate.yaml', 'r'))
ANCHOR_NUM          = GENERATE_YAML['anchor_num']
ANCHOR_POSITIONS    = [GENERATE_YAML[f'anchor{idx}'] for idx in range(ANCHOR_NUM)]
TAG_X_RANGE         = GENERATE_YAML['tag']['x_range']
TAG_Y_RANGE         = GENERATE_YAML['tag']['y_range']
TAG_Z_RANGE         = GENERATE_YAML['tag']['z_range']
TAG_STEP            = GENERATE_YAML['tag']['step']
ERR_STDEV           = GENERATE_YAML['err_stdev']
ERR_MEAN            = GENERATE_YAML['err_mean']
TRIAL_NUM           = GENERATE_YAML['trial_num']

def calcDistance(tag_pos: list, anchor_pos: list) -> float:
    t = tag_pos
    a = anchor_pos
    return np.sqrt((a[0] - t[0])**2.0 + (a[1] - t[1])**2.0 + (a[2] - t[2])**2.0)

def calcDistanceWithNoise(tag_pos: list, anchor_pos: list) -> float:
    t = tag_pos
    a = anchor_pos
    sign1 = 1.0 if (np.random.rand() < 0.5) else -1.0
    sign2 = 1.0 if (np.random.rand() < 0.5) else -1.0
    return np.sqrt((a[0] - t[0])**2.0 + (a[1] - t[1])**2.0 + (a[2] - t[2])**2.0) + sign1*np.random.normal(0.0, ERR_STDEV) + sign2*ERR_MEAN*np.random.rand()

tag_pos_x = np.arange(TAG_X_RANGE[0], TAG_X_RANGE[1], TAG_STEP)
tag_pos_y = np.arange(TAG_Y_RANGE[0], TAG_Y_RANGE[1], TAG_STEP)
tag_pos_z = np.arange(TAG_Z_RANGE[0], TAG_Z_RANGE[1], TAG_STEP)

actual_range = {'no': [], 'tag_x': [], 'tag_y': [], 'tag_z': []}
noised_range = {'no': [], 'tag_x': [], 'tag_y': [], 'tag_z': []}
for idx in range(ANCHOR_NUM):
    actual_range.update({f'anchor{idx}': []})
    noised_range.update({f'anchor{idx}': []})

data_cnt = 0
for tx in tag_pos_x:
    for ty in tag_pos_y:
        for tz in tag_pos_z:
            tag_pos = [tx, ty, tz]
            for i in range(TRIAL_NUM):
                actual_range['no'].append(data_cnt)
                actual_range['tag_x'].append(tx)
                actual_range['tag_y'].append(ty)
                actual_range['tag_z'].append(tz)
                noised_range['no'].append(data_cnt)
                noised_range['tag_x'].append(tx)
                noised_range['tag_y'].append(ty)
                noised_range['tag_z'].append(tz)
                data_cnt += 1
                for ai in range(ANCHOR_NUM):
                    anchor_pos = [ANCHOR_POSITIONS[ai][0], ANCHOR_POSITIONS[ai][1], ANCHOR_POSITIONS[ai][2]]
                    actual_range[f'anchor{ai}'].append(calcDistance(tag_pos, anchor_pos))
                    noised_range[f'anchor{ai}'].append(calcDistanceWithNoise(tag_pos, anchor_pos))

folder_name = f'{uuid.uuid4()}'[:8]
os.mkdir(folder_name)

with open(f'./{folder_name}/actual.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(actual_range.keys())
    writer.writerows(zip(*actual_range.values()))

with open(f'./{folder_name}/read.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(noised_range.keys())
    writer.writerows(zip(*noised_range.values()))

with open(f'./{folder_name}/generate.yaml', 'w') as yamlfile:
    yaml.safe_dump(GENERATE_YAML, yamlfile)