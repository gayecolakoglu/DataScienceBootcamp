
# GOREV 1
# 1
import pandas as pd
df = pd.read_csv('datasets/persona.csv')
df.head()
df.shape
df.info()

# 2
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# 3
df["PRICE"].nunique()

# 4
df["PRICE"].value_counts()

# 5
df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()

# 6
df.groupby("COUNTRY")["PRICE"].sum()

# 7
df["SOURCE"].value_counts()

# 8
df.groupby("COUNTRY")["PRICE"].mean()

# 9
df.groupby("SOURCE")["PRICE"].mean()

# 10
df.groupby(by=["COUNTRY", "SOURCE"]).agg({"PRICE":"mean"})

################################################################################

# GOREV 2
new_df = df.groupby(by=["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"})
new_df.head()

# GOREV 3
new_df = new_df.sort_values("PRICE", ascending=False)
new_df.head()

# GOREV 4
new_df = new_df.reset_index()
new_df.head()

# GOREV 5
points = [0, 10, 18, 28, 40, new_df["AGE"].max()]
names = ["0_10", "11_18", "19_28", "29_40", "41_" + str(new_df["AGE"].max())]
new_df["cat_age"] = pd.cut(new_df["AGE"], points, labels=names)
new_df.head()

# GOREV 6
new_df["customer_level_based"] = [i[0].upper() + "_" + i[1].upper() +
                                  "_" + i[2].upper() + "_" + i[5].upper()
                                  for i in new_df.values]

new_df.head()

new_df = new_df[["customer_level_based", "PRICE"]]
new_df.head()

for i in new_df["customer_level_based"].values:
    print(i.split("_"))

new_df["customer_level_based"].value_counts()
new_df = new_df.groupby("customer_level_based").agg({"PRICE":"mean"})
new_df = new_df.reset_index()
new_df.head()

new_df["customer_level_based"].value_counts()
new_df.head()

# GOREV 7
new_df["SEGMENT"] = pd.qcut(new_df["PRICE"], 4, labels=["D","C", "B","A"])
new_df.head()
new_df.groupby("SEGMENT").agg({"PRICE":"mean"})

# GOREV 8
new_customer = "TUR_ANDROID_FEMALE_29_40"
new_df[new_df["customer_level_based"] == new_customer]

