import pandas as pd
import numpy as np 
from scipy import stats
from scipy.stats import norm, skew 


def mytime(tik_local):
    from time import time
    tak = time()
    print('all time is - ', tak-tik, '(sec)')
    print('local time is - ', tak-tik_local, '(sec)')
    return tak
#procedure start example of def mytime
#tik_local = mytime(tik_local)


#Unique Data Analysis
#процедура аналицирует количество одинаковых значений в столбце и если их больше чем unique_part*100%,
#то создает список на удаление
def unique_analysis(unique_part):
    col_for_drop = []
    for col in all_data.columns:
        if (np.max(all_data[col].value_counts())/len(all_data[col]))>unique_part:
            print(col,np.max(all_data[col].value_counts()), ' of ', len(all_data[col]), ' per cent is ', 
                  np.max(all_data[col].value_counts())*100//len(all_data[col]),'%')
            col_for_drop.append(col)
    print
    return col_for_drop #возвращает список столбцов таблицы с количеством большим чем unique_part
#procedure start example of def unique_analysis
#col_for_drop = unique_analysis(0.8)


#Missing Data analysis
def missing_analysis(all_data):
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio (% of total all_data)' :all_data_na})
    print(missing_data.head(20))
#procedure start example of def unique_analysis
#missing_analysis(all_data)


#determinate train to numeric and categoric data
# Take analysis of numeric and categorical columns
def list_columns(all_data):
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    start_len_num = len(numeric_feats)
    print('len of the numeric_feats is: ', start_len_num)
    categorical_feats = all_data.dtypes[all_data.dtypes == "object"].index
    start_len_cat = len(categorical_feats)
    print('len of the categorical_feats is: ', start_len_cat)
    print('total len of all_data_feats is: ', len(all_data.columns))
    return numeric_feats, categorical_feats


#other def for determing the numeric and categiric features (columns of dataframe) as lists
def columns_len(all_data):
    cat_cols = [c for c in all_data.columns if all_data[c].dtype.name == 'object']
    num_cols   = [c for c in all_data.columns if all_data[c].dtype.name != 'object']
    return num_cols, cat_cols
#procedure start example of def list_columns
#num_cols, cat_cols = list_columns(all_data)

#******************************************************************************    
def num_col_plus_minus(all_data):
    num_cols, cat_cols = columns_len(all_data)
    for col in num_cols:
        for coli in num_cols:
            if col != coli:
                str1 = col + '_plus_' + coli
                str2 = col + '_minus_' + coli
                all_data[str1] = all_data[col] + all_data[coli]
                all_data[str2] = all_data[col] - all_data[coli]
                #print(col, '---', coli)    


# Определение вылетов по межквартильному размаху в столбцах с цифрами
# Помечаем вылеты через дополнительные столбцы col +'Outlier'
#создаем функцию, которая возвращает индекс выбросов
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3-q1
    lower_bond = q1 - (iqr*1.5)
    upper_bond = q3 + (iqr*1.5)
    return np.where((x > upper_bond) | (x < lower_bond))
#основная функция
# делаем дополнительные бинарные столбы с признаком выброса по каждому числовому столбцу
def outlier_mark_percentile(all_data):
    num_cols, cat_cols = columns_len(all_data)
    for col in num_cols:
        pp = indicies_of_outliers(all_data[col])
        list_index = pp[0]
        if len(list_index)>1:
            name_col = col +'Outlier'
            all_data[name_col] = 0
            all_data[name_col][list_index] = 1
            print(len(list_index), ' --- ', all_data[name_col].sum(), ' --- ', name_col)
#procedure start example of outlier_mark_percentile
#outlier_mark_percentile(all_data)        



# класстеризация поля '1stFlrSF', '2ndFlrSF' на 5 групп через метод к-средних
# и запись распределение данных по группам в новую колонку

def kmean_clust(dataframe, col1, col2, num_groups):  #dataframe, col1, col2 as string, num_groups as int
    from sklearn.cluster import KMeans
    dataframe = dataframe[[col1, col2]]
    #создаем кластеризатор по методу К средних
    clusterer = KMeans(num_groups, random_state = 0)
    #выполнить подгонку кластеризатора
    clusterer.fit(dataframe)
    #предсказать значения в новую колонку
    dataframe[col1+col2+'kmean_clust'] = clusterer.predict(dataframe)


#Снижение размерности с помощью выделения признаков (метод главных компонент)
def pca_features(all_data):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    num_cols, cat_cols = columns_len(all_data)
    features = StandardScaler().fit_transform(all_data[num_cols])
    pca = PCA(n_components=0.99, whiten = True)
    features_pca = pca.fit_transform(features)
    print('исходное количество признаков:', features.shape[1])
    print('сокращенное количество признаков (pca):', features_pca.shape[1])


#Уменьшение количества признаков путем максимизации разделимости классов
# до количества столбцов n_components, остальные считаются менее важными
def cat_features(all_data, target, n_components):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    num_cols, cat_cols = columns_len(all_data)
    features = all_data[num_cols]
    target = target
    lda = LinearDiscriminantAnalysis(n_components =n_components)
    features_lda = lda.fit(features, target).transform(features)
    print('исходное количество признаков:', features.shape[1])
    print('сокращенное количество признаков lda:', features_lda.shape[1])


#уменьшения количества признаков на разряженных матрицах
# надо доделать после one-hot coding на категорийных столбцах 
def feat_unrise_onehot(all_data, target, n_components):    #n_components as int
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import csr_matrix
    num_cols, cat_cols = columns_len(all_data)
    features = all_data[num_cols]
    target = target
    features = StandardScaler().fit_transform(features)
    feature_sparse = csr_matrix(features)
    tsvd = TruncatedSVD(n_components = n_components)
    features_sparse_tsvd = tsvd.fit(feature_sparse).transform(feature_sparse)
    print('исходное количество признаков:', features.shape[1])
    print('сокращенное количество признаков lda:', features_sparse_tsvd.shape[1])    


#пороговая обработка дисперсии числовых признаков
#в этом методе признаки не должны быть стандартизованы иначе у них у всех будет одинаковая дисперсия
#суть метода в том, что высокодисперсные признаки несут больше информации чем низкодисперсные
def feat_threshold(all_data, target):
    from sklearn.feature_selection import VarianceThreshold
    num_cols, cat_cols = columns_len(all_data)
    features = all_data[num_cols]
    target = target
    thresholder = VarianceThreshold(threshold = .5)
    features_high_variance = thresholder.fit_transform(features)
    print('исходное количество признаков:', features.shape[1])
    print('сокращенное количество признаков lda:', features_high_variance.shape[1])
    #print(features_high_variance[:5])
    #взглянуть на дисперсию отобранных признаков
    print(thresholder.fit(features).variances_)


def corr_data(all_data, target):
    num_cols, cat_cols = columns_len(all_data)
    features = all_data[num_cols]
    target = target    
    #обработка высококоррелированных признаков
    #создаем корреляционную матрицу
    corr_matrix = features.corr().abs()
    #выбираем верхний треугольник корреляционной матрицы (т.к. нижний симметричный)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #находим индекс столбцов признаком с корреляцией больше 0.95
    to_drop = [column for column in upper.columns if any(upper[column]>0.95)]
    print(to_drop)
    # исключаем эти признаки
    features.drop(to_drop, axis =1).head(3)
    print('количество признаков:', features.shape[1])



#удаление нерелевантных признаков для классификации
def kbest_cat(all_data, target):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, f_classif
    num_cols, cat_cols = columns_len(all_data)
    features = all_data[cat_cols]
    target = target
    #for cаtegoric features
    #конвертировать категориальные данные в целые числа
    features = features.astype(int)
    #отобрать два признака с наивысшими значениями 
    #статистического показателя хи-квадрат
    chi2_selector = SelectKBest(chi2, k=2)
    features_kbest = chi2_secetor.fit_transform(features, target)

def kbest_num(all_data, target):
    num_cols, cat_cols = columns_len(all_data)
    features = all_data[num_cols]
    target = target
    #for nuneric features
    #отобрать два признака с наивысшими значениями 
    #статистического показателя F
    fvalue_selector = SelectKBest(f_classif, k=2)
    features_kbest = fvalue_selector.fit_transform(features, target)

def kbest_all(all_data, target):
    #for all types features
    from sklearn.feature_selection import SelectPercentile
    #отобрать верхние 75% признаков с наивысшими значениями 
    #статистического показателя F
    fvalue_selector = SelectPercentile(f_classif, percentile = 75)
    features_kbest = fvalue_selector.fit_transform(all_data, target)

#**********************************************
def skewed_features(all_data):
    #Skewed features
    # Check the skew of all numerical features
    skewed_feats = all_data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(10)
    #Box Cox Transformation of (highly) skewed features
    skewness = skewness[abs(skewness) > 0.75]    # My_DS_work - was 0.75 my vers - 0.5
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)
    #all_data[skewed_features] = np.log1p(all_data[skewed_features])
    #Getting dummy categorical features
    all_data = pd.get_dummies(all_data)
    print(all_data.shape)    
    

def rfecv_features(all_data, target, num_ranking):
    #рекурсивное устранение признаков после нормирования признаков через бокс-кокс
    # num_kanking - номер ранжирование признака от самого лучшего (1) до самого плохого, integer
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LinearRegression
    import warnings
    warnings.filterwarnings(action='ignore', module = 'scipy', message='^internal gelsd')
    num_cols, cat_cols = columns_len(all_data)
    features = all_data[num_cols]    
    ols = LinearRegression()
    rfecv = RFECV(estimator=ols, step=1, scoring='neg_mean_squared_error')
    rfecv.fit(features, target)
    rfecv.transform(features)
    print(rfecv.n_features_, 'количество наилучших признаков')
    print("какие категории самые лучшие: \n", rfecv.support_)
    print("ранжирование признака от самого лучшего (1) до самого плохого: \n", rfecv.ranking_)
    ii_col = pd.concat([pd.DataFrame(all_data.columns, columns=['feature']), pd.DataFrame(rfecv.ranking_, columns=['ranking'])], axis=1)
    ii_col = ii_col.sort_values(by=['ranking'])
    best_features = ii_col[ii_col['ranking']<num_ranking]['feature']
    #print(ii_col)
    print(best_features)
    return best_features   #as list


# *********************  MODELS **********************************
#neg_mean_squared_error for models
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

#
#procedure start example of outlier_mark_percentile
#score = rmsle_cv(lasso)
#str_lasso = "\nLasso score: {:.4f} ({:.4f})".format(score.mean(), score.std())
#print(str_lasso)