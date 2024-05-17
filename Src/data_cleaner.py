import numpy as np
import pandas as pd

def clean_bib(bib):

    return int(bib.strip('F'))

def seconds_converter(time):
    if time == '-':
        return np.nan

    time_split = [int(x) for x in (time.split(':'))]
    return time_split[0] * (60 ** 2) + time_split[1] * 60 + time_split[2]


def best_fit_slope(x, y):
    m = (((x.mean() * y.mean()) - (x * y).mean()) /
         (x.mean() ** 2 - (x ** 2).mean()))
    return round(m, 2)


def race_slope(row):
    nK_splits = ['5K', '10K', '15K', '20K', '25K', '30K', '35K', '40K']
    x = np.array([5, 10, 15, 20, 25, 30, 35, 40])
    y = np.array([row['5K']] + list(np.diff(row[nK_splits])))
    return best_fit_slope(x, y)


def race_slope_alternative(row):

    if np.isnan(row['pace_rate']):
        half_splits = ['Half', 'finish_time']
        x = np.array([21.1, 42.2])
        y = np.array([row['Half']] + list(np.diff(row[half_splits])))
        return best_fit_slope(x, y) / 4
    return row['pace_rate']


def add_weather_data(df, year):

    year = int(year)
    median = df['finish_time'].median() / 3600

    pre_median_2015 = {'temp': 48, 'humidity': 0.71, 'wind': 17}
    post_median_2015 = {'temp': 44, 'humidity': 0.89, 'wind': 16}
    pre_median_2016 = {'temp': 64, 'humidity': 0.29, 'wind': 14}
    post_median_2016 = {'temp': 53, 'humidity': 0.55, 'wind': 14}
    pre_median_2017 = {'temp': 74, 'humidity': 0.29, 'wind': 21}
    post_median_2017 = {'temp': 73, 'humidity': 0.24, 'wind': 22}
    pre_median_2018 = {'temp': 45, 'humidity': 0.93, 'wind': 14}
    post_median_2018 = {'temp': 44, 'humidity': 1.00, 'wind': 17}

    new_cols = ['temp', 'humidity', 'wind']

    for col in new_cols:
        if year == 2015:
            df[col] = [pre_median_2015[col] if x < median else post_median_2015[col] for x in df['finish_time']]
        elif year == 2016:
            df[col] = [pre_median_2016[col] if x < median else post_median_2016[col] for x in df['finish_time']]
        elif year == 2017:
            df[col] = [pre_median_2017[col] if x < median else post_median_2017[col] for x in df['finish_time']]
        else:
            df[col] = [pre_median_2018[col] if x < median else post_median_2018[col] for x in df['finish_time']]

    return df


def overall_ratio(row, max_rank):
    return row['overall_rank'] / max_rank


def gender_ratio(row, fmax, mmax):
    if row['Gender_F'] == 1:
        return row['gender_rank'] / fmax
    return row['gender_rank'] / mmax


def division_ratio(row, max1, max2, max3, max4, max5, max6, max7, max8, max9, max10):
    if row['Age'] < 40:
        return row['division_rank'] / max1
    if row['Age'] >= 40 and row['Age'] < 45:
        return row['division_rank'] / max2
    if row['Age'] >= 45 and row['Age'] < 50:
        return row['division_rank'] / max3
    if row['Age'] >= 50 and row['Age'] < 55:
        return row['division_rank'] / max4
    if row['Age'] >= 55 and row['Age'] < 60:
        return row['division_rank'] / max5
    if row['Age'] >= 60 and row['Age'] < 65:
        return row['division_rank'] / max6
    if row['Age'] >= 65 and row['Age'] < 70:
        return row['division_rank'] / max7
    if row['Age'] >= 70 and row['Age'] < 75:
        return row['division_rank'] / max8
    if row['Age'] >= 75 and row['Age'] < 80:
        return row['division_rank'] / max9
    if row['Age'] >= 80:
        return row['division_rank'] / max10


def rank_ratio(df):
    overall_max_rank = df['overall_rank'].max()
    df['overall_rank'] = df.apply(overall_ratio, max_rank=overall_max_rank, axis=1)

    female_max = df[df['Gender_F'] == 1]['gender_rank'].max()
    male_max = df[df['Gender_M'] == 1]['gender_rank'].max()
    df['gender_rank'] = df.apply(gender_ratio, axis=1, fmax=female_max, mmax=male_max)

    division1_max = df[df['Age'] < 40]['division_rank'].max()
    division2_max = df[(df['Age'] >= 40) & (df['Age'] < 45)]['division_rank'].max()
    division3_max = df[(df['Age'] >= 45) & (df['Age'] < 50)]['division_rank'].max()
    division4_max = df[(df['Age'] >= 50) & (df['Age'] < 55)]['division_rank'].max()
    division5_max = df[(df['Age'] >= 55) & (df['Age'] < 60)]['division_rank'].max()
    division6_max = df[(df['Age'] >= 60) & (df['Age'] < 65)]['division_rank'].max()
    division7_max = df[(df['Age'] >= 65) & (df['Age'] < 70)]['division_rank'].max()
    division8_max = df[(df['Age'] >= 70) & (df['Age'] < 75)]['division_rank'].max()
    division9_max = df[(df['Age'] >= 75) & (df['Age'] < 80)]['division_rank'].max()
    division10_max = df[df['Age'] >= 80]['division_rank'].max()
    df['division_rank'] = df.apply(division_ratio, axis=1, max1=division1_max,
                                   max2=division2_max, max3=division3_max,
                                   max4=division4_max, max5=division5_max,
                                   max6=division6_max, max7=division7_max,
                                   max8=division8_max, max9=division9_max,
                                   max10=division10_max)
    return df

def clean_data(df, year):



    df.drop_duplicates(['Bib'], inplace=True)


    df['Bib'] = df['Bib'].apply(clean_bib)


    df.rename(columns={'Overall': 'overall_rank', 'Gender': 'gender_rank', 'Official Time': 'finish_time',
                       'Division': 'division_rank', 'M/F': 'Gender'}, inplace=True)


    time_cols = ['5K', '10K', '15K', '20K', 'Half', '25K',
                 '30K', '35K', '40K', 'finish_time']
    for col in time_cols:
        df[col] = df[col].apply(seconds_converter)

    df['pace_rate'] = df.apply(race_slope, axis=1)
    df['pace_rate'] = df.apply(race_slope_alternative, axis=1)
    df['pace_rate'].fillna(df['pace_rate'].mean(), inplace=True)


    df = add_weather_data(df, year)

    df = pd.get_dummies(df, columns=['Gender'])

    df = rank_ratio(df)

    df.drop(['Citizen', 'City', 'State', 'Country', '5K', '10K', '15K', '20K',
             '25K', '30K', '35K', '40K', 'Pace', 'Half'], axis=1, inplace=True)
    df.drop(list(df.filter(regex='Unnamed')), axis=1, inplace=True)
    df.drop(list(df.filter(regex='Proj')), axis=1, inplace=True)

    return df

def merge_dataframes(dfA, dfB, legacy_runners):

    dfA_legacy = dfA[dfA['Name'].isin(legacy_runners)]
    dfB_legacy = dfB[dfB['Name'].isin(legacy_runners)]

    df_merge = dfA.merge(dfB, how='left', on='Name', suffixes=('_A', '_B'))

    df_merge = df_merge[df_merge['Age_A'] == df_merge['Age_B'] - 1]

    drop_cols = ['finish_time_A', 'Bib_B', 'Age_B', 'overall_rank_B', 'gender_rank_B',
                 'division_rank_B', 'pace_rate_B', 'temp_B', 'humidity_B', 'wind_B',
                 'Gender_F_B', 'Gender_M_B']
    df_merge.drop(drop_cols, inplace=True, axis=1)
    df_merge.columns = ['Bib', 'Name', 'Age', 'overall_rank', 'gender_rank',
                          'division_rank', 'pace_rate', 'temp', 'humidity',
                          'wind', 'Gender_F', 'Gender_M', 'finish_time']
    return df_merge
