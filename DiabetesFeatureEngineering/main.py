import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sns.get_dataset_names()
df= pd.read_csv("../../DataScienceBootcamp/DiabetesFeatureEngineering/diabetes.csv")


def general_idea(df):
    print("----HEAD:")
    print(df.head())
    print("\n")
    print("----SHAPE:")
    print(df.shape)
    print("\n")
    print("----TYPES:")
    print(df.dtypes)
    print("\n")
    print("----NULL:")
    print(df.isnull().sum())
    print("\n")
    print("----IQR:")
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

general_idea(df)

def grab_col_names(df, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if
                   df[col].nunique() < cat_th and df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > car_th and df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(df, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100*df[col_name].value_counts()/len(df)}))

    print("*******************************")
    if plot:
        sns.countplot(x=df[col_name], data=df)
        plt.show()

cat_summary(df,"Outcome")


def num_summary(df, num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[num_cols].describe(quantiles).T)

    if plot:
        df[num_cols].hist(bins=20)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

def target_summary_with_num(df, target, num_col):
    print(df.groupby(target).agg({num_col:"mean"}),end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

df.corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#BASE MODEL

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"AUC: {round(roc_auc_score(y_pred, y_test), 2)}")

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.Dataframe({'Value': model.feature_importanece_, 'Feature':features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Vallue", y='Feature', data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])

    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)

#NULL VALUES
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ['Pregnancies', 'Outcome'])]
zero_columns

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

df.isnull().sum()

def missing_values_table(df, na_name=False):
    na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_cols].isnull().sum() / df.shape[0]*100).sort_values(ascending=False)
    df_missing = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(df_missing, end="\n")
    if na_name:
        return na_cols

na_cols = missing_values_table(df, na_name=True)

def missing_vs_target(Df, target, na_cols):
    df_temp = df.copy()
    for col in na_cols:
        df_temp[col + '_NA_FLAG'] = np.where(df_temp[col].isnull(), 1, 0)
    na_flags = df_temp.loc[:, df_temp.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.Dataframe({"TARGET_MEAN": df_temp.gropby(col)[target].mean(),
                            "Count": df_temp.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_cols)

#filling na values
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()

#OUTLİERS
def outlier_th(df, col_name, q1=0.05, q3=0.95):
    quart1 = df[col_name].quantile(q1)
    quart3 = df[col_name].quantile(q3)
    iqr = quart3 - quart1
    up_limit = quart3 + 1.5*iqr
    low_limit = quart1 - 1.5*iqr
    return low_limit, up_limit

def check_outliers(df, col_name):
    low_limit, up_limit = outlier_th(df, col_name)
    if df[(df[col_name] > up_limit) | (df[col_name] > low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_th(df, var, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_th(df, var)
    df.loc[(df[var] < low_limit), var] = low_limit
    df.loc[(df[var] > up_limit), var] = up_limit

for col in df.columns:
    print(col, check_outliers(df, col))
    if check_outliers(df, col):
        replace_with_th(df, col)

for col in df.columns:
    print(col, check_outliers(df, col))


#FEATURE EXTRACTION
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 0.18, 24.9, 29.9, 100],
                       labels=["Underweight", "Healthy", "Overweight", "Obese"])

df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

df.loc[(df["BMI"] < 18.5) & ((df["Age"] > 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (
    (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 2)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (
    (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 50)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"


df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (
    (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 150) & (df["Glucose"] <= 125)) & (
    (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 150) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 50) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

def set_insulin(df, col_name="Insulin"):
    if 16 <= df[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)
df["NEW_GLUCOSE*INSULIN"] = df["Glucose"]*df["Insulin"]
df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"]*df["Pregnancies"]


df.columns = [col.upper() for col in df.columns]

df.head()
df.shape

#ENCODING
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#--LabelEncoding
def label_encoder(df, binary_cols):
    labelencoder = LabelEncoder()
    df[binary_cols] = labelencoder.fit_transform(df[binary_cols])
    return df

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

#--OneHotEncoding
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(df, cat_cols, drop_first=False):
    df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape
df.head()

#STANDARDAZITAON
num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape

#MODEL
y = df["OUTCOME"]
x = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"AUC: {round(roc_auc_score(y_pred, y_test), 2)}")

#FEATUre IMPORTANCE
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.Dataframe({'Value': model.feature_importanece_, 'Feature':features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Vallue", y='Feature', data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])

    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)