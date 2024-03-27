import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("bengaluru_house_prices.csv")
df =pd.DataFrame(df)
print(df.head(10))
print(df.shape)
print()
print(df.groupby("area_type")["area_type"].aggregate("count"))

df2 = df.drop(["availability","area_type","society","balcony"],axis="columns")
df2 = pd.DataFrame(df2)
print(df2.head(10))
print()
print(df2.isnull().sum())
print()

df3 = df2.dropna()
df3 = pd.DataFrame(df3)
print(df3.isnull().sum())
print()
print(df3["size"].unique())
df3["bhk"] = df3["size"].apply(lambda x: int(x.split(" ")[0]))
print()
print(df3.head(10))
print(df3["bhk"].unique())
print()
print(df3[df3.bhk>20])
print()
print(df3["total_sqft"].unique())
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
print()
print(df3[~df3['total_sqft'].apply(is_float)].head(20))
def convert_sqrt_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return ((float(tokens[0])+float(tokens[1]))/2)
    try:
        return float(x)
    except:
        return None
    
df4 = df3.copy()
df4["total_sqft"] = df4["total_sqft"].apply(convert_sqrt_to_num)
print(df4.loc[30])

df5 = df4.copy()
df5["price_per_sqft"] = (df5["price"]*100000)/df5["total_sqft"]
print()
print(df5.head(10))
print()
print(len(df5.location.unique()))
df5.location =df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby("location")["location"].agg("count").sort_values(ascending=False)
print(location_stats)
print()
print(len(location_stats[location_stats<=10]))
location_stats_less_than_10 = location_stats[location_stats<=10]
df5.location = df5.location.apply(lambda x: "other" if x in location_stats_less_than_10 else x )
print(df5.head(10))
print()
print()
print(df5[df5.total_sqft/df5.bhk < 300].head(20))

df6 =df5[~(df5.total_sqft/df5.bhk < 300)]
print()
print(df6.price_per_sqft.describe())
def remove_pps_outliers(df):
    df_out =pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df =subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out, reduced_df],ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)
print()
print(df7.shape)
def plot_scatter_chart(df , location):
    bhk2 =df[(df.location == location) & (df.bhk==2)]
    bhk3 =df[(df.location == location) & (df.bhk==3)]
    matplotlib.rcParams["figure.figsize"] = (15,10)
    plt.scatter(bhk2.total_sqft , bhk2.price, color="blue" ,label ="2 BHK",s=50)
    plt.scatter(bhk3.total_sqft , bhk3.price ,color="green" ,marker ="+" ,label="3 BHK" ,s=50)
    plt.xlabel("total square feet area")
    plt.ylabel("price per square feet")
    plt.title(location)
    plt.legend()
    plt.show()
#plot_scatter_chart(df7, "Rajaji Nagar")
#plot_scatter_chart(df7, "Hebbal")
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                "mean" : np.mean(bhk_df.price_per_sqft),
                "std"  : np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk-1)
            if stats and stats["count"]>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats["mean"])].index.values)
    return df.drop(exclude_indices, axis="index")
    
df8 = remove_bhk_outliers(df7)
print()
print(df8.shape)
#plot_scatter_chart(df8, "Rajaji Nagar")
#plot_scatter_chart(df8, "Hebbal")
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft , rwidth= 0.8)
plt.xlabel("price per square feet")
plt.ylabel("count")
#plt.show()
print()
print(df8.bath.unique())
print(df8[df8.bath > 10])
plt.hist(df8.bath , rwidth= 0.8)
plt.xlabel("number of bathrooms")
plt.ylabel("count")
#plt.show()
print()
print(df8[df8.bath > df8.bhk+2])

df9 =df8[df8.bath < df8.bhk+2]
print()
print(df9.shape)

df10 = df9.drop(["size", "price_per_sqft"],axis="columns")
print()
print(df10.head())

dummies = pd.get_dummies(df10.location)
print(dummies.head(10))

df11 = pd.concat([df10,dummies.drop("other",axis="columns")],axis="columns")
print(df11.head(10))

df12 = df11.drop("location",axis="columns")
print(df12.head(10))