# GOREV 1

# 1
import pandas as pd
df = pd.read_excel("datasets/miuul_gezinomi.xlsx")
df.head()

# 2
df["SaleCityName"].nunique()
df["SaleCityName"].value_counts()

# 3
df["ConceptName"].nunique()

# 4
df["ConceptName"].value_counts()

# 5
df.groupby("SaleCityName").agg({"Price":"sum"})

# 6
df.groupby("ConceptName").agg({"Price":"sum"})

# 7
df.groupby(by=["SaleCityName"]).agg({"Price":"mean"})

# 8
df.groupby(by=["ConceptName"]).agg({"Price":"mean"})

# 9
df.groupby(by=["SaleCityName", "ConceptName"]).agg({"Price":"mean"})

##############################################################

# GOREV 2

bins = [-1, 7, 30, 90, df["SaleCheckInDayDiff"].max()]
labels = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]

df["EB_Scroe"] = pd.cut(df["SaleCheckInDayDiff"], bins, labels=labels)
df.head(50).to_excel("eb_scorew.xlsx", index=False)

###############################################################

# GOREV 3
df.groupby(by=["SaleCityName", "ConceptName", "EB_Score"]).agg({"Price":["mean", "count"]})

df.groupby(by=["SaleCityName", "ConceptName", "Seasons"]).agg({"Price":["mean", "count"]})

df.groupby(by=["SaleCityName", "ConceptName", "CInDay"]).agg({"Price":["mean", "count"]})

###################################################################

# GOREV 4
new_df = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price":"mean"}).sort_values("Price", ascending=False)
new_df.head()

###################################################################

# GOREV 5
new_df.reset_index(inplace=True)
new_df.head()

###################################################################

# GOREV 6
new_df["sales_level_based"] = new_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '_'.join(x).upper(), axis=1)

###################################################################

# GOREV 7
new_df["SEGMENT"] = pd.qcut(new_df["Price"], 4, labels=["D", "C", "B", "A"])
new_df.head()
new_df.groupby("SEGMENT").agg({"Price":["mean", "max", "sum"]})

###################################################################

# GOREV 8
new_df.sort_values(by="Price")

new_customer = "ANTALYA_HERŞEY_DAHİL_HİGH"
new_df[new_df["sales_level_based"] == new_customer]

