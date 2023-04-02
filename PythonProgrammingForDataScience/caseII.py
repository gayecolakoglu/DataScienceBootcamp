import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns
df.info()

# GÖREV 1:
["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

# GÖREV 2:
[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]

# GÖREV 3:
og_list = ["abbrev", "no_previous"]
new_cols = [ col for col in df.columns if col not in og_list]
new_df = df[new_cols]
new_df.head()

### PANDAS ###

# GÖREV 1:
df = sns.load_dataset("titanic")
df.head()
df.shape

# GÖREV 2:
df["sex"].value_counts()

# GÖREV 3:
df.nunique()

# GÖREV 4:
df["pclass"].unique()

# GÖREV 5:
df[["pclass","parch"]].nunique()

# GÖREV 6:
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype

# GÖREV 7:
df[df["embarked"]=="C"].head()

# GÖREV 8:
df[df["embarked"]!="S"].head()

# GÖREV 9:
df[(df["age"]<30) & (df["sex"]=="female")].head()

# GÖREV 10:
df[(df["fare"]>500) | (df["age"]>70)].head()

# GÖREV 11:
df.isnull().sum()

# GÖREV 12:
df.drop("who", axis=1, inplace=True)

# GÖREV 13:
df["deck"].mode()
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull()
df["deck"].isnull().sum()

# GÖREV 14:
df["age"].median()
df["age"].fillna(df["age"].median(), inplace=True)
df["age"].isnull()
df["age"].isnull().sum()

# GÖREV 15:
df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})

# GÖREV 16:
def new_age(age):
    if age<30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x : new_age(x))

# GÖREV 17:
df = sns.load_dataset("tips")
df.head()

# GÖREV 18:
df.groupby("time").agg({"total_bill":["sum", "min", "max", "mean"]})

# GÖREV 18:
df.groupby(["time","day"]).agg({"total_bill":["sum", "min", "max", "mean"]})

# GÖREV 20:
df[ (df["time"]=="Lunch") & (df["sex"]=="Female") ].groupby("day").agg(
{"total_bill":["sum", "min", "max", "mean"],
 "tip":["sum", "min", "max", "mean"]}
)

# GÖREV 21:
df.loc[(df["size"]<3) & (df["total_bill"]>10), "total_bill"].mean()

# GÖREV 22:
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

# GÖREV 23:
new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.head()
new_df.shape