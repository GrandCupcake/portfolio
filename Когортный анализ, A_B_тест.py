#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from operator import attrgetter


# In[4]:


#Задание № 1


# In[5]:


file_path = "/mnt/HC_Volume_18315164/home-jupyter/jupyter-aleksandr-ladejsch-6e962/shared/problem1-reg_data.csv"


# In[6]:


file_path2 = "/mnt/HC_Volume_18315164/home-jupyter/jupyter-aleksandr-ladejsch-6e962/shared/problem1-auth_data.csv"


# In[7]:


reg_data=pd.read_csv(file_path,sep=';')
auth_data=pd.read_csv(file_path2,sep=';')


# In[51]:


reg_data.head()


# In[52]:


reg_data.dtypes


# In[48]:


reg_data.isnull().sum()


# In[49]:


reg_data.duplicated().sum()


# In[53]:


auth_data.head()


# In[54]:


auth_data.dtypes


# In[58]:


auth_data.isnull().sum()


# In[61]:


auth_data.duplicated().sum()


# In[ ]:





# In[40]:



def cohort_analysis(reg_data, auth_data, 
                   start_date=None, end_date=None
                   ):
  
    reg_data=pd.read_csv(file_path,sep=';')
    auth_data=pd.read_csv(file_path2,sep=';')
    
    reg_data["reg_ts"] = pd.to_datetime(reg_data["reg_ts"], origin='unix', unit="s")
    auth_data["auth_ts"] = pd.to_datetime(auth_data["auth_ts"], origin='unix', unit="s")
    
    
    df_merged = pd.merge(reg_data, auth_data, how='inner', on='uid')
    
   
    if start_date:
        df_merged = df_merged[df_merged['reg_ts'] >= start_date]
    if end_date:
        df_merged = df_merged[df_merged['reg_ts'] <= end_date]
    
   
    df_merged['cohort'] = df_merged['reg_ts'].dt.to_period('D')
    df_merged['auth_period'] = df_merged['auth_ts'].dt.to_period('D')
    
   
    df_merged['period'] = (df_merged['auth_period'] - df_merged['cohort']).apply(attrgetter('n'))
    
   
    cohort_data = df_merged.groupby(['cohort', 'period']).agg(
        n_users=('uid', 'nunique')
    ).reset_index()
    
   
    cohort_pivot = cohort_data.pivot_table(
        index='cohort',
        columns='period',
        values='n_users',
        aggfunc='sum'
    ).fillna(0)
    

    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100
    if 0 in retention_matrix.columns:
        retention_matrix = retention_matrix.drop(0, axis=1)
    
   
    return retention_matrix


# In[41]:


retention=cohort_analysis(reg_data, auth_data, start_date= '2020-09-07', end_date='2020-09-22')


# In[42]:


retention


# In[ ]:


#Задание № 2


# In[65]:


import requests
from urllib.parse import urlencode
from scipy.stats import ttest_ind, chi2_contingency

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/SOkIsD5A8xlI7Q'  


final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']


download_response = requests.get(download_url)
with open('Проект_1_Задание_2.csv', 'wb') as f:   
    f.write(download_response.content)


# In[66]:


df=pd.read_csv('Проект_1_Задание_2.csv',sep=';')


# In[67]:


control_group=df.query('testgroup == "a"')


# In[68]:


test_group=df.query('testgroup == "b"')


# In[69]:


ARPU_test = test_group['revenue'].sum() / test_group['user_id'].nunique()
ARPU_control = control_group['revenue'].sum() / control_group['user_id'].nunique()
ARPU_dif = (ARPU_test - ARPU_control) / ((ARPU_test + ARPU_control)/2)  * 100
print(f"ARPU тестовой группы: {ARPU_test:.2f}")
print(f"ARPU контрольной группы: {ARPU_control:.2f}")
print(f"Разница в ARPU : {ARPU_dif:.2f} %")


# ARPU тестовой группы больше на 5.13 % .

# In[70]:


n_control = len(control_group)
n_test = len(test_group)
payers_control = (control_group['revenue'] > 0).sum()
payers_test = (test_group['revenue'] > 0).sum()


# In[71]:


cr_control = payers_control / n_control * 100
cr_test = payers_test / n_test * 100


# In[72]:


print(f"Conversion Rate контрольной группы: {cr_control:.2f} %")
print(f"Conversion Rate тестовой группы: {cr_test:.2f} %")


# Конверсия в платящих пользователей  контрольной группы больше чем конверсия в платящих пользователей тестовой группы.
# 

# In[73]:


ARPPU_control=control_group['revenue'].sum()/payers_control
ARPPU_test=test_group['revenue'].sum()/payers_test


# In[74]:


print(f"ARPPU тестовой группы: {ARPPU_test:.2f}")
print(f"ARPPU контрольной группы: {ARPPU_control:.2f}")


# В тестовой группе платящие пользователи тратят больше чем в контрольной группе.

# Обоснование использования статистических тестов:
# при достаточно больших выборках можно полагаться на центральную предельную теорему, которая гарантирует, что распределение средних будет стремиться к нормальному. Это позволяет использовать t‑тест для сравнения средних значений, даже если исходные данные не распределены нормально.

# In[85]:


t_stat_arpu, p_val_arpu = ttest_ind(control_group['revenue'], test_group['revenue'], equal_var=False)


# In[86]:


print(f"t-статистика ARPU: {t_stat_arpu}")
print(f"p-value ARPU: {p_val_arpu}")


# In[98]:


revenue_control_paying = control_group[control_group["revenue"] > 0]["revenue"]
revenue_test_paying = test_group[test_group["revenue"] > 0]["revenue"]
t_stat_arppu, p_val_arppu = ttest_ind(revenue_control_paying, revenue_test_paying, equal_var=False)


# In[99]:


print(f"t-статистика ARPPU: {t_stat_arppu}")
print(f"p-value ARPPU: {p_val_arppu}")


# In[92]:


print(f"t-статистика ARPPU: {t_stat_arppu}")
print(f"p-value ARPPU: {p_val_arppu}")


# p-value ARPU > 0.05 , что выше порогового значения. Разница статистически незначима.
# 
# для ARPPU тоже нет статистически значимых различий.

#          Платящие    Неплатящие
# Контроль          1928         200175
# Тест              1805         200862

# In[143]:


contingency_table = [
    [payers_control, n_control - payers_control],
    [payers_test, n_test - payers_test]
]
chi2_stat, chi2_p_val,_,_= chi2_contingency(contingency_table)


# In[144]:


contingency_table


# In[146]:


print(f"Статистика хи-квадрат CR: {chi2_stat}")
print(f"p-value хи-квадрат CR: {chi2_p_val}")


# p-value хи-квадрат CR меньше 0.05  . Разница в конверсии значима.

# Результат: Статистически значимого увеличения ARPU или ARPPU в тестовой группе зафиксировано не было, тогда как конверсия в контрольной группе оказалась статистически значимо выше. Это говорит о том, что тестируемый набор предложений нельзя считать однозначно успешным, и его не рекомендуется внедрять для всей аудитории.

# In[ ]:


# 3 Задание 


# Базовое событие:
# 
# 1)DAU (Daily Active Users) и MAU (Monthly Active Users)
# 
# 2)Event Participation Rate – % активных игроков, принявших участие в событии
# 
# 3)First-Time Entry – сколько игроков запустили событие хотя бы раз
# 
# 4)Event Revenue – выручка, полученная в рамках события
# 
# 5)Conversion Rate – % игроков, сделавших покупку ради прохождения события
# 
# 6)ARPU / ARPPU по событию
# 
# 7)Event Retention  – удержание участников события
# 
# Усложнение механики влияет на поведение и восприятие события игроками. Это требует расширения и переоценки набора метрик:
# 
# 1)Retry Rate per Level – сколько попыток в среднем требуется на уровень
# 
# 2)Failure Rate – % неудачных попыток на каждом уровне
# 
# 3)Rollback Trigger Rate – % попыток, приведших к откату
# 
# 4)Average Rollback Depth – на сколько уровней в среднем откатывается игрок
# 
# 5)Frustration Proxy Metrics – резкий рост выходов из игры после отката, паузы, уменьшение времени сессий
# 
# 6)Drop-off после отката – % игроков, завершивших участие сразу после отката
# 
# 7)Complaint / Support Rate – обращения в поддержку или негатив в отзывах
# 
# Если игроки уходят из события сразу после откатов или не возвращаются на следующий день — это тревожный сигнал.
# 
