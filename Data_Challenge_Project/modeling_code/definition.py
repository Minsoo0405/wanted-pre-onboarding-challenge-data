import numpy as np

def rmsle(y,pred):
    actual_values = np.array(y)
    predicted_values = np.array(pred)

    log_actual = np.log(actual_values + 1)
    log_predict = np.log(predicted_values + 1)

    difference = (log_predict - log_actual) ** 2

    mean_difference = difference.mean()
    score = np.sqrt(mean_difference)

    return score

def remove_outlier(df,col):
    # "Rent Bike Count" 데이터에서 전체의 25%에 해당하는 데이터 조회
    count_q1 = np.percentile(df[col], 25)
    print(f"Q1 : {count_q1}")

    # "Rent Bike Count" 데이터에서 전체의 75%에 해당하는 데이터 조회
    count_q3 = np.percentile(df[col], 75)
    print(f"Q3 : {count_q3}")

    # IQR = Q3 - Q1
    count_IQR = count_q3 - count_q1
    print(f"IQR : {count_IQR}")

    # 이상치를 제외한(이상치가 아닌 구간에 있는) 데이터만 조회
    train_df = df[(df[col] >= (count_q1 - (1.5 * count_IQR))) & (df[col] <= (count_q3 + (1.5 * count_IQR)))]
    print("Normal Data Shape: ", train_df.shape)

    test_df = df[(df[col] < (count_q1 - (1.5 * count_IQR))) | (df[col] > (count_q3 + (1.5 * count_IQR)))]
    print("Outlier Shape: ", test_df.shape)

    return train_df, test_df