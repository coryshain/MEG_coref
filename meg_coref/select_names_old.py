import pandas as pd

df = pd.read_csv('data/MEG/nyc_names_12-23-20.csv')
df["Child's First Name"] = df["Child's First Name"].str.capitalize()
df['Ethnicity'] = df['Ethnicity'].apply(
    lambda x: 'ASIAN AND PACIFIC ISLANDER' if x == 'ASIAN AND PACI'
    else 'BLACK NON HISPANIC' if x == 'BLACK NON HISP'
    else 'WHITE NON HISPANIC' if x == 'WHITE NON HISP'
    else x
)

df = df.sort_values("Child's First Name")
df = df.drop_duplicates("Child's First Name")

df = df.sort_values(['Gender', 'Ethnicity', 'Rank'])
df['Index'] = df.groupby(['Gender', 'Ethnicity']).cumcount()
df = df[df.Index < 100]

df.to_csv('output/stim/names_sorted.csv', index=False)