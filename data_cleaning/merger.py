import pandas as pd

feat_df = pd.read_csv('combined.csv', index_col=0)
df = pd.read_csv('readmes.csv', index_col=0)
dd = pd.get_dummies(df)
dd['subject'] = dd.index.str[1:].astype(int)

dd = dd[['age', 'height', 'weight', 'gender_ female', 'gender_ male',
                   'coffee_today_YES', 'sport_today_YES', 'smoker_NO', 'smoker_YES',
                   'feel_ill_today_YES', 'subject']]

m = pd.merge(feat_df, dd, on='subject')
m.to_csv('final.csv')
