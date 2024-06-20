import numpy as np

from tqdm.notebook import tqdm
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score


def forward_backward(X_train, X_test, y_train, y_test, features_shap, model, type_selection='backward', stop=False):
    
    f''' Функция получающая на вход: 
    1. Набор заданных на обучении параметров X_train, X_test, y_train, y_test     (считалось при обучении на train_test_split или посчитать дополнительно)
    2. Список предикторов, отсортированный по важности, где первый - бесполезный, последний - важнейший (обычно по shap, можно по дефлтному importance)  
    3. Модель с уже вставленными параметрами, для которой мы будем отбрасывать предикторы
    4. Тип отбора фичей. По дефолту='backward', для прямого прохождения ставим 'forward'
    5. stop - ограничение по количеству фичей (по_дефолту=False, если задаем, то числом остающихся (не больше количества фичей))
    '''
    
    info = {
        'feature' : [],
        'num_features' : [],
        'mean_gini_cv_valid' : [],
        'std_gini_cv_valid' : [],
        'mean_gini_cv_train' : [],
        'std_gini_cv_train' : [],
        'gini_train' : [],
        'gini_test' : [],
        'overfit' : []
    }
    
    if type_selection == 'backward':                                      
        
        constant_features_lst = features_shap.copy()                        # перезадаем список фичей для перебора
        features_ = set(constant_features_lst.copy())                       # стартовое количество фичей
        adjustment = 1                                                      # корректировка по длине перебора для backward (необязательно, но так красивее)
    
    elif type_selection == 'forward':
        
        constant_features_lst = features_shap[::-1].copy()
        features_ = set()
        adjustment = 0
    
    else:
        return 'Неправильно указан тип обработки'
    
    # ---------------------- Цикл обработки ------------------------------
    
    for feature in tqdm(constant_features_lst, total=len(constant_features_lst) - adjustment ):     
        
        
        if type_selection == 'backward': 
        
            features_ = list(set(features_)- set([feature]))                   # убираем из списка фичей для обучения ненужную фичу
    
            if not features_ or (stop and len(features_) < stop):              # когда убрали до нуля или проставили порог - останавливаемся
                break
        else: # forward
            
            features_ = list(set(features_) | set([feature]))                  # объединяем множество features_ и текущую фичу
    
            if not features_ or (stop and len(features_) > stop):              # если фичей вдруг нет или проставили порог - останавливаемся
                break
            
        # ------------- кросс валидация ----------------    
        cv_data = cross_validate(                           # смотрим 10 или сколько то (cv) кроссвалидационных скоров
            model, 
            X_train[features_], 
            y_train,
            scoring='roc_auc',
            return_train_score=True, 
            cv=6, 
            #error_score='raise'
            )
        valid_cv_gini = 2 * cv_data['test_score'] - 1      # считаем np.array значений gini для тестовых выборок
        train_cv_gini = 2 * cv_data['train_score'] - 1     # считаем np.array значений gini для трейн выборок

        # ------------ обучение и расчет параметров -------------
        
        model.fit(
            X_train[features_], 
            y_train, 
            #cat_features=cat_features_, 
            eval_set=(X_test[features_], y_test), 
            plot=False)
    
        roc_auc_train = roc_auc_score(y_train, model.predict_proba(X_train[features_])[:, 1])
        roc_auc_test = roc_auc_score(y_test, model.predict_proba(X_test[features_])[:, 1])
    
        gini_train = 2 * roc_auc_train - 1
        gini_test = 2 * roc_auc_test - 1
        
        # ---------------------Запоминаем ------------------------
    
        info['feature'].append(feature)                                    # имя отброшенной фичи
        info['num_features'].append(len(features_))                        # длина оставшегося количества фичей
        info['mean_gini_cv_valid'].append(np.mean(valid_cv_gini))          # среднее по тестовым джини на кросс валидации для текущего кол-ва фичей
        info['std_gini_cv_valid'].append(np.std(valid_cv_gini))            # стандартное отклонение по тестовым джини на кросс валидации для текущего кол-ва фичей
        info['mean_gini_cv_train'].append(np.mean(train_cv_gini))          # среднее по трейн джини на кросс валидации для текущего кол-ва фичей
        info['std_gini_cv_train'].append(np.std(train_cv_gini))            # стандартное отклонение по трейн джини на кросс валидации для текущего кол-ва фичей
        info['gini_train'].append(gini_train)                              # джини трейн для данного количества фичей
        info['gini_test'].append(gini_test)                                # джини тест для данного количества фичей
        info['overfit'].append(gini_train - gini_test)                     # оверфит для данного количества фичей
        
    return info