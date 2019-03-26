from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from catboost import CatBoostClassifier
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc


'''
Feature Engineering

Для значений, начинающихся с запятой:
c_te[cc] = pd.to_numeric(c_te[cc].apply(lambda x: re.sub(',', '.', str(x))))

количество потребленных услуг каждым абонентом в месяц

MTBD (Mean Time Beetwen Drops)
'''

pd.options.display.width = None
pd.options.mode.chained_assignment = None


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\nTime taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def load_data():
    print('Load Dataset')
    subs_csi_train = pd.read_csv('dataset/train/subs_csi_train.csv', sep=';')
    print('subs_csi_train loaded')
    #subs_features_train = pd.read_csv('dataset/train/subs_features_train.csv', sep=';')
    #print('subs_features_train loaded')
    subs_bs_consumption_train = pd.read_csv('dataset/train/subs_bs_consumption_train.csv', sep=';')
    print('subs_bs_consumption_train loaded')
    #subs_bs_data_session_train = pd.read_csv('dataset/train/subs_bs_data_session_train.csv', sep=';')
    #subs_bs_voice_session_train = pd.read_csv('dataset/train/subs_bs_voice_session_train.csv', sep=';')

    subs_csi_test = pd.read_csv('dataset/test/subs_csi_test.csv', sep=';')
    print('subs_csi_test loaded')
    #subs_features_test = pd.read_csv('dataset/test/subs_features_test.csv', sep=';')
    #print('subs_features_test loaded')
    subs_bs_consumption_test = pd.read_csv('dataset/test/subs_bs_consumption_test.csv', sep=';')
    print('subs_bs_consumption_test loaded')
    #subs_bs_data_session_test = pd.read_csv('dataset/test/subs_bs_data_session_test.csv', sep=';')
    #subs_bs_voice_session_test = pd.read_csv('dataset/test/subs_bs_voice_session_test.csv', sep=';')

    #bs_avg_kpi = pd.read_csv('dataset/bs_avg_kpi.csv', sep=';')
    #bs_chnn_kpi = pd.read_csv('dataset/bs_chnn_kpi.csv', sep=';')
    return subs_csi_train, subs_bs_consumption_train, subs_csi_test, subs_bs_consumption_test


def load_data2():
    subs_features_train = pd.read_csv('dataset/train/subs_features_train.csv', sep=';')
    print('subs_features_train loaded')
    subs_features_test = pd.read_csv('dataset/test/subs_features_test.csv', sep=';')
    print('subs_features_test loaded')
    return subs_features_train, subs_features_test


def apply_agregation(df: pd.DataFrame, agregation: str):
    name = ''
    if agregation == 'sum':
        name = '_SUM'
    elif agregation == 'mean':
        name = '_MEAN'
    elif agregation == 'median':
        name = '_MEDIAN'
    elif agregation == 'mad':
        name = '_MAD'
    elif agregation == 'min':
        name = '_MIN'
    elif agregation == 'max':
        name = '_MAX'
    elif agregation == 'prod':
        name = '_PROD'
    elif agregation == 'std':
        name = '_STD'
    elif agregation == 'var':
        name = '_VAR'

    df.rename(columns={'COM_CAT#1': str('COM_CAT#1' + name),
                       'COM_CAT#2': str('COM_CAT#2' + name),
                       'COM_CAT#3': str('COM_CAT#3' + name),
                       'BASE_TYPE': str('BASE_TYPE' + name),
                       'ACT': str('ACT' + name),
                       'ARPU_GROUP': str('ARPU_GROUP' + name),
                       'COM_CAT#7': str('COM_CAT#7' + name),
                       'COM_CAT#8': str('COM_CAT#8' + name),
                       'DEVICE_TYPE_ID': str('DEVICE_TYPE_ID' + name),
                       'INTERNET_TYPE_ID': str('INTERNET_TYPE_ID' + name),
                       'REVENUE': str('REVENUE' + name),
                       'ITC': str('ITC' + name),
                       'VAS': str('VAS' + name),
                       'RENT_CHANNEL': str('RENT_CHANNEL' + name),
                       'ROAM': str('ROAM' + name),
                       'COST': str('COST' + name),
                       'COM_CAT#17': str('COM_CAT#17' + name),
                       'COM_CAT#18': str('COM_CAT#18' + name),
                       'COM_CAT#19': str('COM_CAT#19' + name),
                       'COM_CAT#20': str('COM_CAT#20' + name),
                       'COM_CAT#21': str('COM_CAT#21' + name),
                       'COM_CAT#22': str('COM_CAT#22' + name),
                       'COM_CAT#23': str('COM_CAT#23' + name),
                       'COM_CAT#24': str('COM_CAT#24' + name),
                       'COM_CAT#25': str('COM_CAT#25' + name),
                       'COM_CAT#26': str('COM_CAT#26' + name),
                       'COM_CAT#27': str('COM_CAT#27' + name),
                       'COM_CAT#28': str('COM_CAT#28' + name),
                       'COM_CAT#29': str('COM_CAT#29' + name),
                       'COM_CAT#30': str('COM_CAT#30' + name),
                       'COM_CAT#31': str('COM_CAT#31' + name),
                       'COM_CAT#32': str('COM_CAT#32' + name),
                       'COM_CAT#33': str('COM_CAT#33' + name),
                       'COM_CAT#34': str('COM_CAT#34' + name)}, inplace=True)
    return df


def processing_subs_consumption(df: pd.DataFrame):
    print('Processing subs_consumption')
    df = df[['SK_ID', 'SUM_DATA_MB', 'SUM_DATA_MIN', 'SUM_MINUTES']]
    df.loc[:, 'SUM_DATA_MB'] = [x.replace(',', '.') for x in df.loc[:, 'SUM_DATA_MB']]
    df.loc[:, 'SUM_DATA_MB'] = df.loc[:, 'SUM_DATA_MB'].astype(float)
    df.loc[:, 'SUM_DATA_MIN'] = [x.replace(',', '.') for x in df.loc[:, 'SUM_DATA_MIN']]
    df.loc[:, 'SUM_DATA_MIN'] = df.loc[:, 'SUM_DATA_MIN'].astype(float)
    df.loc[:, 'SUM_MINUTES'] = [x.replace(',', '.') for x in df.loc[:, 'SUM_MINUTES']]
    df.loc[:, 'SUM_MINUTES'] = df.loc[:, 'SUM_MINUTES'].astype(float)
    #print(df)
    #df = df.stack().str.replace(',', '0.').unstack()
    df_sum = df.groupby(['SK_ID'], as_index=False).sum()
    df_sum.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_SUM',
                           'SUM_DATA_MIN': 'SUM_DATA_MIN_SUM',
                           'SUM_MINUTES': 'SUM_MINUTES_SUM'}, inplace=True)
    df_mean = df.groupby(['SK_ID'], as_index=False).mean()
    df_mean.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_MEAN',
                            'SUM_DATA_MIN': 'SUM_DATA_MIN_MEAN',
                            'SUM_MINUTES': 'SUM_MINUTES_MEAN'}, inplace=True)
    df_median = df.groupby(['SK_ID'], as_index=False).median()
    df_median.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_MEDIAN',
                              'SUM_DATA_MIN': 'SUM_DATA_MIN_MEDIAN',
                              'SUM_MINUTES': 'SUM_MINUTES_MEDIAN'}, inplace=True)
    df_mad = df.groupby(['SK_ID'], as_index=False).mad()
    df_mad.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_MAD',
                           'SUM_DATA_MIN': 'SUM_DATA_MIN_MAD',
                           'SUM_MINUTES': 'SUM_MINUTES_MAD'}, inplace=True)
    df_min = df.groupby(['SK_ID'], as_index=False).min()
    df_min.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_MIN',
                           'SUM_DATA_MIN': 'SUM_DATA_MIN_MIN',
                           'SUM_MINUTES': 'SUM_MINUTES_MIN'}, inplace=True)
    df_max = df.groupby(['SK_ID'], as_index=False).max()
    df_max.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_MAX',
                           'SUM_DATA_MIN': 'SUM_DATA_MIN_MAX',
                           'SUM_MINUTES': 'SUM_MINUTES_MAX'}, inplace=True)
    df_prod = df.groupby(['SK_ID'], as_index=False).prod()
    df_prod.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_PROD',
                            'SUM_DATA_MIN': 'SUM_DATA_MIN_PROD',
                            'SUM_MINUTES': 'SUM_MINUTES_PROD'}, inplace=True)
    df_std = df.groupby(['SK_ID'], as_index=False).std()
    df_std.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_STD',
                           'SUM_DATA_MIN': 'SUM_DATA_MIN_STD',
                           'SUM_MINUTES': 'SUM_MINUTES_STD'}, inplace=True)
    df_var = df.groupby(['SK_ID'], as_index=False).var()
    df_var.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_VAR',
                           'SUM_DATA_MIN': 'SUM_DATA_MIN_VAR',
                           'SUM_MINUTES': 'SUM_MINUTES_VAR'}, inplace=True)
    df_sem = df.groupby(['SK_ID'], as_index=False).sem()
    df_sem.rename(columns={'SUM_DATA_MB': 'SUM_DATA_MB_SEM',
                           'SUM_DATA_MIN': 'SUM_DATA_MIN_SEM',
                           'SUM_MINUTES': 'SUM_MINUTES_SEM'}, inplace=True)
    '''
    print('sum rows:', len(df_sum))
    print('sum rows:', len(df_mean))
    print('sum rows:', len(df_median))
    print('sum rows:', len(df_mad))
    print('sum rows:', len(df_min))
    print('sum rows:', len(df_max))
    print('sum rows:', len(df_prod))
    print('sum rows:', len(df_std))
    print('sum rows:', len(df_var))
    print('sum rows:', len(df_sem))
    '''
    df_temp = merge_data(df_A=df_sum, df_B=df_mean)
    df_temp = merge_data(df_A=df_temp, df_B=df_median)
    df_temp = merge_data(df_A=df_temp, df_B=df_mad)
    df_temp = merge_data(df_A=df_temp, df_B=df_min)
    df_temp = merge_data(df_A=df_temp, df_B=df_max)
    df_temp = merge_data(df_A=df_temp, df_B=df_prod)
    df_temp = merge_data(df_A=df_temp, df_B=df_std)
    df_temp = merge_data(df_A=df_temp, df_B=df_var)
    #df_temp = merge_data(df_A=df_temp, df_B=df_sem)
    print('processed rows:', len(df_temp))
    return df_temp


def processing_subs_features(df: pd.DataFrame):
    print('Processing subs_features')
    #df = df.drop(['SNAP_DATE', 'COM_CAT#24'], axis=1)
    df = df[['SK_ID',
             'COM_CAT#1',
             'COM_CAT#2',
             'COM_CAT#3',
             'BASE_TYPE',
             'ACT',
             'ARPU_GROUP',
             'COM_CAT#7',
             'COM_CAT#8',
             'DEVICE_TYPE_ID',
             'INTERNET_TYPE_ID',
             'REVENUE',
             'ITC',
             'VAS',
             'RENT_CHANNEL',
             'ROAM',
             'COST',
             'COM_CAT#17',
             'COM_CAT#18',
             'COM_CAT#19',
             'COM_CAT#20',
             'COM_CAT#21',
             'COM_CAT#22',
             'COM_CAT#23',
             'COM_CAT#25',
             'COM_CAT#26',
             'COM_CAT#27',
             'COM_CAT#28',
             'COM_CAT#29',
             'COM_CAT#30',
             'COM_CAT#31',
             'COM_CAT#32',
             'COM_CAT#33',
             'COM_CAT#34']]
    df.loc[:, 'REVENUE'] = [x.replace(',', '.') for x in df.loc[:, 'REVENUE']]
    df.loc[:, 'REVENUE'] = df.loc[:, 'REVENUE'].astype(float)
    df.loc[:, 'ITC'] = [x.replace(',', '.') for x in df.loc[:, 'ITC']]
    df.loc[:, 'ITC'] = df.loc[:, 'ITC'].astype(float)
    df.loc[:, 'VAS'] = [x.replace(',', '.') for x in df.loc[:, 'VAS']]
    df.loc[:, 'VAS'] = df.loc[:, 'VAS'].astype(float)
    df.loc[:, 'RENT_CHANNEL'] = [x.replace(',', '.') for x in df.loc[:, 'RENT_CHANNEL']]
    df.loc[:, 'RENT_CHANNEL'] = df.loc[:, 'RENT_CHANNEL'].astype(float)
    df.loc[:, 'ROAM'] = [x.replace(',', '.') for x in df.loc[:, 'ROAM']]
    df.loc[:, 'ROAM'] = df.loc[:, 'ROAM'].astype(float)
    df.loc[:, 'COST'] = [x.replace(',', '.') for x in df.loc[:, 'COST']]
    df.loc[:, 'COST'] = df.loc[:, 'COST'].astype(float)
    df.loc[:, 'COM_CAT#17'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#17']]
    df.loc[:, 'COM_CAT#17'] = df.loc[:, 'COM_CAT#17'].astype(float)
    df.loc[:, 'COM_CAT#18'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#18']]
    df.loc[:, 'COM_CAT#18'] = df.loc[:, 'COM_CAT#18'].astype(float)
    df.loc[:, 'COM_CAT#19'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#19']]
    df.loc[:, 'COM_CAT#19'] = df.loc[:, 'COM_CAT#19'].astype(float)
    df.loc[:, 'COM_CAT#20'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#20']]
    df.loc[:, 'COM_CAT#20'] = df.loc[:, 'COM_CAT#20'].astype(float)
    df.loc[:, 'COM_CAT#21'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#21']]
    df.loc[:, 'COM_CAT#21'] = df.loc[:, 'COM_CAT#21'].astype(float)
    df.loc[:, 'COM_CAT#22'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#22']]
    df.loc[:, 'COM_CAT#22'] = df.loc[:, 'COM_CAT#22'].astype(float)
    df.loc[:, 'COM_CAT#23'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#23']]
    df.loc[:, 'COM_CAT#23'] = df.loc[:, 'COM_CAT#23'].astype(float)
    df.loc[:, 'COM_CAT#27'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#27']]
    df.loc[:, 'COM_CAT#27'] = df.loc[:, 'COM_CAT#27'].astype(float)
    df.loc[:, 'COM_CAT#28'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#28']]
    df.loc[:, 'COM_CAT#28'] = df.loc[:, 'COM_CAT#28'].astype(float)
    df.loc[:, 'COM_CAT#29'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#29']]
    df.loc[:, 'COM_CAT#29'] = df.loc[:, 'COM_CAT#29'].astype(float)
    df.loc[:, 'COM_CAT#30'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#30']]
    df.loc[:, 'COM_CAT#30'] = df.loc[:, 'COM_CAT#30'].astype(float)
    df.loc[:, 'COM_CAT#31'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#31']]
    df.loc[:, 'COM_CAT#31'] = df.loc[:, 'COM_CAT#31'].astype(float)
    df.loc[:, 'COM_CAT#32'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#32']]
    df.loc[:, 'COM_CAT#32'] = df.loc[:, 'COM_CAT#32'].astype(float)
    df.loc[:, 'COM_CAT#33'] = [x.replace(',', '.') for x in df.loc[:, 'COM_CAT#33']]
    df.loc[:, 'COM_CAT#33'] = df.loc[:, 'COM_CAT#33'].astype(float)

    df_sum = df.groupby(['SK_ID'], as_index=False).sum()
    df_sum = apply_agregation(df_sum, 'sum')
    df_mean = df.groupby(['SK_ID'], as_index=False).mean()
    df_mean = apply_agregation(df_mean, 'mean')
    df_median = df.groupby(['SK_ID'], as_index=False).median()
    df_median = apply_agregation(df_median, 'median')
    df_mad = df.groupby(['SK_ID'], as_index=False).mad()
    df_mad = apply_agregation(df_mad, 'mad')
    df_min = df.groupby(['SK_ID'], as_index=False).min()
    df_min = apply_agregation(df_min, 'min')
    df_max = df.groupby(['SK_ID'], as_index=False).max()
    df_max = apply_agregation(df_max, 'max')
    df_prod = df.groupby(['SK_ID'], as_index=False).prod()
    df_prod = apply_agregation(df_prod, 'prod')
    df_std = df.groupby(['SK_ID'], as_index=False).std()
    df_std = apply_agregation(df_std, 'std')
    df_var = df.groupby(['SK_ID'], as_index=False).var()
    df_var = apply_agregation(df_var, 'var')
    '''
    print('sum rows:', len(df_sum))
    print('mean rows:', len(df_mean))
    print('median rows:', len(df_median))
    print('mad rows:', len(df_mad))
    print('min rows:', len(df_min))
    print('max rows:', len(df_max))
    print('prod rows:', len(df_prod))
    print('std rows:', len(df_std))
    print('var rows:', len(df_var))
    '''
    df_temp = merge_data(df_A=df_sum, df_B=df_mean)
    df_temp = merge_data(df_A=df_temp, df_B=df_median)
    df_temp = merge_data(df_A=df_temp, df_B=df_mad)
    df_temp = merge_data(df_A=df_temp, df_B=df_min)
    df_temp = merge_data(df_A=df_temp, df_B=df_max)
    df_temp = merge_data(df_A=df_temp, df_B=df_prod)
    df_temp = merge_data(df_A=df_temp, df_B=df_std)
    df_temp = merge_data(df_A=df_temp, df_B=df_var)
    print('processed rows:', len(df_temp))
    return df_temp

def merge_data(df_A: pd.DataFrame, df_B: pd.DataFrame):
    #print('Merging dataframes')
    df_C = pd.merge(df_A, df_B, on='SK_ID', how='left')
    return df_C


def join_data(df_A: pd.DataFrame, df_B: pd.DataFrame):
    print('Merging dataframes')
    #df_C = pd.join(other.set_index('key'), on='key')


def catboost_clf(X_train, y_train, X_test):
    print('Catboost Classifier')
    model = CatBoostClassifier(iterations=10000, random_state=42)
    model.fit(X=X_train, y=y_train, verbose=200)
    preds_class = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)
    preds_class = pd.DataFrame(preds_class)
    return preds_class


def xgboost_clf(X_train, y_train, X_test):
    print('XGBoost Classifier')
    params = {'nthread': 10, 'max_depth': 9, 'n_estimators': 3000, 'subsample': 0.9, 'gamma': 0.1,
              'learning_rate': 0.01, 'seed': 0, 'random_state': 42, 'predictor': 'cpu_predictor', 'eval_metric': 'auc'}
    model = XGBClassifier(silent=False, **params)
    model.fit(X=X_train, y=y_train)
    preds_class = model.predict(X_test)
    #preds_proba = model.predict_proba(X_test)
    preds_class = pd.DataFrame(preds_class)
    return preds_class


def submit(preds: pd.DataFrame):
    name = 'clf_catboost.csv'
    preds.to_csv(name, header=None, index=None)
    print('Submit Ok')


def drop_features(X_train, X_test):
    columns_train = ['CONTACT_DATE', 'SK_ID', 'CSI']
    columns_test = ['CONTACT_DATE', 'SK_ID']
    X_train = X_train.drop(columns_train, axis=1)
    X_test = X_test.drop(columns_test, axis=1)
    return X_train, X_test


def main():
    print('Telecom Data Cup - CSI Analyze')
    start_time = timer(None)

    subs_csi_train, subs_bs_consumption_train, subs_csi_test, subs_bs_consumption_test = load_data()
    subs_features_train, subs_features_test = load_data2()

    print('subs_csi_train rows:', len(subs_csi_train))
    print('subs_bs_consumption_train rows:', len(subs_bs_consumption_train))
    print('subs_csi_test rows:', len(subs_csi_test))
    print('subs_bs_consumption_test rows:', len(subs_bs_consumption_test))

    df_consumption_train = processing_subs_consumption(subs_bs_consumption_train)
    X_train = merge_data(df_A=subs_csi_train, df_B=df_consumption_train)

    df_consumption_test = processing_subs_consumption(subs_bs_consumption_test)
    X_test = merge_data(df_A=subs_csi_test, df_B=df_consumption_test)

    df_features_train = processing_subs_features(subs_features_train)
    df_features_test = processing_subs_features(subs_features_test)

    X_train = merge_data(df_A=X_train, df_B=df_features_train)
    X_test = merge_data(df_A=X_test, df_B=df_features_test)

    y_train = X_train['CSI']
    print('Train rows:', len(X_train))
    print('X_train:', X_train.head(5))
    print('Test rows:', len(X_test))
    print('X_test:', X_test.head(5))
    X_train, X_test = drop_features(X_train=X_train, X_test=X_test)

    X_train.to_csv('X_train.csv', header=True, index=None, sep=';')
    X_test.to_csv('X_test.csv', header=True, index=None, sep=';')

    #preds_class = catboost_clf(X_train=X_train, y_train=y_train, X_test=X_test)
    preds_class = xgboost_clf(X_train=X_train, y_train=y_train, X_test=X_test)
    submit(preds_class)

    timer(start_time)
    print('Ok')


if __name__ == '__main__':
    main()

