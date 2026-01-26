# data/config.py

import os

# 1. 任务与标签映射
TASK_CONFIG = {
    'task1': {
        'name': 'informative',
        'num_classes': 2,
        'label_map': {
            'informative': 1,
            'not_informative': 0
        }
    },
    'task2': {
        'name': 'humanitarian',
        'num_classes': 8, # 对应 Full Task
        'label_map': {
            'infrastructure_and_utility_damage': 0,
            'not_humanitarian': 1,
            'other_relevant_information': 2,
            'rescue_volunteering_or_donation_effort': 3,
            'vehicle_damage': 4,
            'affected_individuals': 5,
            'injured_or_dead_people': 6,
            'missing_or_found_people': 7,
        }

    },
    'task3': {
            'name': 'damage',  # 对应文件名 task_damage_text_img_train.tsv
            'num_classes': 3,
            'label_map': {
                'severe_damage': 0,
                'mild_damage': 1,
                'little_or_no_damage': 2
            }
    }
}

# 2. 默认路径配置
# 假设你的数据还在原来的位置，或者你可以把 datasets 文件夹移动过来
DEFAULT_DATA_ROOT = '../datasets/CrisisMMD_v2.0'