import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import re
import matplotlib.pyplot as plt

df = pd.read_excel("retailInvestor_basket_0428.xlsx")
# df.head()
# df.info()
# df.describe()
df_copy = df.copy()
# df_copy.head()
col_name = df_copy.columns
# col_name
col_name = col_name.drop('Symbol Name')
for i in col_name:

    m_buy = df_copy[i] > 100000
    m_none = df_copy[i] < 100000

    df_copy.loc[m_buy, i] = 1
    df_copy.loc[m_none, i] = 0

# df_copy.head()

df_copy.drop(['Symbol Name'], inplace=True, axis=1)
# df_copy.head()

df_copy_temp = df_copy == 1
# df_copy_temp.head()

# df_copy_temp.shape

itemset = apriori(df_copy_temp, min_support=0.1, use_colnames=True)
itemset.sort_values(["support"], ascending=False).head(10)

rules_ca = association_rules(itemset, metric="confidence", min_threshold=0.1)
rules_ca = rules_ca[['antecedents', 'consequents',
                     'support', 'confidence', 'lift']]

rules_ca = rules_ca.sort_values('lift', ascending=False)
# rules_ca.head(10)

rules_ca.reset_index(inplace=True)
# rules_ca.head()

rules_ca['antecedents'] = rules_ca['antecedents'].astype('str')
rules_ca['consequents'] = rules_ca['consequents'].astype('str')
rules_ca['antecedents'] = rules_ca['antecedents'].str.replace("frozenset", "")
rules_ca['consequents'] = rules_ca['consequents'].str.replace("frozenset", "")
rules_ca.antecedents = rules_ca.antecedents.apply(''.join).str.replace(
    '[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '')
rules_ca.consequents = rules_ca.consequents.apply(''.join).str.replace(
    '[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '')
rules_ca['ant_con'] = rules_ca[['antecedents', 'consequents']].apply(
    lambda x: ' '.join(x), axis=1)
# rules_ca

ca_support = rules_ca['support'].values
ca_confidence = rules_ca['confidence'].values
ca_lift = rules_ca['lift'].values

plt.figure(figsize=(15, 3))

for i in range(len(ca_support)):
    ca_support[i] = ca_support[i]
    ca_confidence[i] = ca_confidence[i]

plt.subplot(161)
plt.title("support & confidence")
plt.scatter(ca_support, ca_confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence')
# plt.show()


for i in range(len(ca_support)):
    ca_lift[i] = ca_lift[i]
    ca_confidence[i] = ca_confidence[i]

plt.subplot(163)
plt.title("lift & confidence")
plt.scatter(ca_lift, ca_confidence,  alpha=0.5, marker="*")
plt.xlabel('lift')
plt.ylabel('confidence')
# plt.show()


for i in range(len(ca_support)):
    ca_support[i] = ca_support[i]
    ca_confidence[i] = ca_confidence[i]

plt.subplot(165)
plt.title("support & lift")
plt.scatter(ca_support, ca_lift,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('lift')

plt.show()

# 3d로 그리기
# import plotly.express as px
# df = rules_ca
# fig = px.scatter_3d(df, x='support', y='confidence', z='lift', color= 'ant_con')
# fig.show()
