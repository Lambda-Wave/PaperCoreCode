import numpy as np
import pandas as pd
from scipy.stats import spearmanr


#这个文件是用来为选择层级学习率的

import os
import re
def extract_best_miou_from_logs(folder_path):
    all_miou_values = []
    first_seven_miou = []
    best_miou_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)


            with open(file_path, 'r', encoding='utf-8') as file:
                log_data = file.read()


            miou_pattern = r"IoU_mean: ([0-9\.]+)"
            miou_values = [float(match) for match in re.findall(miou_pattern, log_data)]
            all_miou_values.extend(miou_values)


            if len(first_seven_miou) < 7:
                first_seven_miou.extend(miou_values[:7 - len(first_seven_miou)])


            max_miou = max(all_miou_values) if all_miou_values else None

            best_miou_data.append(max_miou)

            all_miou_values = []
            first_seven_miou = []

    a.append(best_miou_data)

def extract_miou_from_logs(folder_path):
    miou_data = []
    file_path_all = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            print(
                file_path
            )
            file_path_all.append(file_path)

            with open(file_path, 'r', encoding='utf-8') as file:
                log_data = file.read()

            miou_pattern = r"IoU_mean: ([0-9\.]+)"
            miou_values = [float(match) for match in re.findall(miou_pattern, log_data)]


            first_seven_miou = miou_values[:7]
            miou_data.append(first_seven_miou)


    for row in zip(*miou_data):
        a.append(list(row))
        # print(f"[{', '.join(map(str, row))}],")
    return file_path_all

#
# log_list = ["Algo/opt1","Algo/opt2","Algo/opt3",
#             "Algo/opt4","Algo/opt5","Algo/opt6",
#             "Algo/opt7","Algo/opt8"]
#             og_list = ["Algo/MSRA/opt8"]
log_list = ["AlgoFindLr/cli/UNet/layer3"]

probabilities = np.array([
[1-2/96, 1-13/96, 1-10/96, 1-5/96, 1-13/96, 1-18/96, 1-35/96],
[1-3/96, 1-15/96, 1-4/96, 1-6/96, 1-16/96, 1-18/96, 1-34/96],
[1-3/96, 1-11/96, 1-5/96, 1-10/96, 1-11/96, 1-17/96, 1-39/96],
[1-1/96, 1-9/96, 1-12/96, 1-11/96, 1-19/96, 1-23/96, 1-21/96],
[1-1/96, 1-9/96, 1-12/96, 1-11/96, 1-19/96, 1-23/96, 1-21/96],
[1-1/96, 1-9/96, 1-8/96, 1-8/96, 1-16/96, 1-20/96, 1-34/96],])


for log_folder in log_list:

    a = []

    file_path_all = extract_miou_from_logs(log_folder)

    data = np.array(a)

    first_seven_miou = data[0:6]

    rank = [0,0,0,0,0,0,0]
    for num_attempts in range(0, 6):

        i = -1

        selected_miou = first_seven_miou[num_attempts]


        rank_selected = np.argsort(-selected_miou) + 1


        if num_attempts ==0:

            rank[rank_selected[-1]-1]+=probabilities[0][0]
            rank[rank_selected[-2] - 1] += probabilities[0][1]
            rank[rank_selected[-3] - 1] += probabilities[0][2]
            rank[rank_selected[-4] - 1] += probabilities[0][3]
            rank[rank_selected[-5] - 1] += probabilities[0][4]
            rank[rank_selected[-6] - 1] += probabilities[0][5]
            rank[rank_selected[-7]-1]+=probabilities[0][6]
        elif num_attempts ==1:
            rank[rank_selected[-1]-1]+=probabilities[1][0]
            rank[rank_selected[-2] - 1] += probabilities[1][1]
            rank[rank_selected[-3] - 1] += probabilities[1][2]
            rank[rank_selected[-4] - 1] += probabilities[1][3]
            rank[rank_selected[-5] - 1] += probabilities[1][4]
            rank[rank_selected[-6] - 1] += probabilities[1][5]
            rank[rank_selected[-7]-1]+=probabilities[1][6]
        elif num_attempts ==2:
            rank[rank_selected[-1]-1]+=probabilities[2][0]
            rank[rank_selected[-2] - 1] += probabilities[2][1]
            rank[rank_selected[-3] - 1] += probabilities[2][2]
            rank[rank_selected[-4] - 1] += probabilities[2][3]
            rank[rank_selected[-5] - 1] += probabilities[2][4]
            rank[rank_selected[-6] - 1] += probabilities[2][5]
            rank[rank_selected[-7]-1]+=probabilities[2][6]

        elif num_attempts ==3:
            rank[rank_selected[-1]-1]+=probabilities[3][0]
            rank[rank_selected[-2] - 1] += probabilities[3][1]
            rank[rank_selected[-3] - 1] += probabilities[3][2]
            rank[rank_selected[-4] - 1] += probabilities[3][3]
            rank[rank_selected[-5] - 1] += probabilities[3][4]
            rank[rank_selected[-6] - 1] += probabilities[3][5]
            rank[rank_selected[-7]-1]+=probabilities[3][6]
        elif num_attempts ==4:
            rank[rank_selected[-1]-1]+=probabilities[4][0]
            rank[rank_selected[-2] - 1] += probabilities[4][1]
            rank[rank_selected[-3] - 1] += probabilities[4][2]
            rank[rank_selected[-4] - 1] += probabilities[4][3]
            rank[rank_selected[-5] - 1] += probabilities[4][4]
            rank[rank_selected[-6] - 1] += probabilities[4][5]
            rank[rank_selected[-7]-1]+=probabilities[4][6]
        elif num_attempts ==5:
            rank[rank_selected[-1]-1]+=probabilities[5][0]
            rank[rank_selected[-2] - 1] += probabilities[5][1]
            rank[rank_selected[-3] - 1] += probabilities[5][2]
            rank[rank_selected[-4] - 1] += probabilities[5][3]
            rank[rank_selected[-5] - 1] += probabilities[5][4]
            rank[rank_selected[-6] - 1] += probabilities[5][5]
            rank[rank_selected[-7]-1]+=probabilities[5][6]

    final_index = np.argsort(rank) + 1


    print(
        "The lr is:"+str(file_path_all[final_index[-1]].split(".")[1].split("train_log")[0])
    )



