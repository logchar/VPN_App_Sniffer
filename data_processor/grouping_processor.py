import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# 将csv文件中的数据根据时间戳分组
def csv_to_group(file_path, time_interval=3.0,time_split=100):
    if time_split==0:
        raise ValueError("time_split 不可为0")
    
    def generate_time_split_list(tmplist:list,time_interval:float,time_split:int):
        """ 
        根据给定的时间间隔和分割数对输入列表进行分组计数。 (该function仅供csv_to_group使用)
        @param tmplist: 需要分组计数的列表。 
        @param time_interval: 总时间间隔。 
        @param time_split: 分割数。 
        @return: 包含每个分组计数结果的列表。 
        """
        time_split_list=[]
        for time in range(time_split):
            timeRoof=(time+1)*(time_interval/time_split)
            timeFloor=time*(time_interval/time_split)
            time_split_list.append(sum(1 for x in tmplist if x >= timeFloor and x <= timeRoof))
        return time_split_list
    
    def time_split_func(need_split:list,time_split_list:list):
        """ 
        根据给定的分组列表对输入列表进行分组求和。 (该function仅供csv_to_group使用)
        @param need_split: 需要分组求和的列表。 
        @param time_split_list: 指定每个分组的元素个数的列表。 
        @return: 包含每个分组求和结果的列表。 
        """
        have_split:int=0
        final_vector=[]
        for item_count in time_split_list:
            final_vector.append(float(sum(x for x in need_split[have_split:have_split+item_count])))
            have_split+=item_count
        return final_vector
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    # 按时间间隔分组
    # 将时间戳除以时间间隔并向下取整，以此作为新的分组键
    min_timestamp = df['timestamp'].iloc[0]
    df['group'] = ((df['timestamp'] - min_timestamp) // time_interval).astype(int)
    grouped = df.groupby('group')

    result = []
    for group_name, group_data in grouped:
        # 分别提取 direction 为 1 和 -1 的数据
        direction_1 = group_data[group_data['direction'] == 1]
        direction_minus_1 = group_data[group_data['direction'] == -1]

        # 创建特征向量
        feature_vector_1 =       time_split_func    (direction_1['length'].tolist(),
                                                     generate_time_split_list(
                                                        (direction_1['timestamp']-min_timestamp-direction_1['group']*time_interval).to_list(),
                                                        time_interval,
                                                        time_split
                                                        )
                                                    )
        feature_vector_minus_1 = time_split_func    (direction_minus_1['length'].tolist(),
                                                     generate_time_split_list(
                                                        (direction_minus_1['timestamp']-min_timestamp-direction_minus_1['group']*time_interval).to_list(),
                                                        time_interval,
                                                        time_split
                                                        )
                                                    )

        result.append({
            'group': group_name,
            'feature_vector_1': feature_vector_1,
            'feature_vector_minus_1': feature_vector_minus_1
        })

    return result


# 每组中两个特征向量的融合
def prepare_data_for_training(processed_data, label):
    X = []
    y = []
    for group in processed_data:
        feature_vector_1 = group['feature_vector_1']
        feature_vector_minus_1 = group['feature_vector_minus_1']
        def formula(x:float,y:float):
            return x+y*2**20
        combined_feature_vector = [formula(x,y) for x,y in zip(feature_vector_1,feature_vector_minus_1)]
        X.append(combined_feature_vector)
        y.append(label)
        print(combined_feature_vector)
    return X, y


# 填充变长的特征向量
def pad_feature_vectors(feature_vectors, max_length):
    padded_vectors = []
    for fv in feature_vectors:
        padding_length = max_length - len(fv)
        padded_vector = fv + [0] * padding_length
        padded_vectors.append(padded_vector)
    return padded_vectors


# 最后的数据处理，之后交给模型训练
def model_data_processor():
    all_X = []
    all_y = []

    # "ike2", "openvpn", "sstp"
    protocals = ["ike2"]
    for ptc in protocals:
        directory_path = "../csv_data/" + str(ptc)
        label = 0
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = directory_path + "/" + file

                processed_data = csv_to_group(file_path)
                X, y = prepare_data_for_training(processed_data, label)
                all_X.extend(X)
                all_y.extend(y)

                label += 1  # 每个文件对应一个应用，每个应用对应一个标签

    # 找到最长的特征向量长度
    max_feature_length = max(len(fv) for fv in all_X)
    # 填充所有特征向量（保证特征向量长度相同，这样才能满足sklearn的需求）
    all_X_padded = pad_feature_vectors(all_X, max_feature_length)

    X_train, X_test, y_train, y_test = train_test_split(all_X_padded, all_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

model_data_processor()

# test grouping
# file_path = '../csv_data/test.csv'
# processed_data = csv_to_group(file_path)
# for entry in processed_data:
#      print(f"Group: {entry['group']}")
#      print(f"Feature Vector (direction 1): {entry['feature_vector_1']}")
#      print(f"Feature Vector (direction -1): {entry['feature_vector_minus_1']}")
#      print()


