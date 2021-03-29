import pandas as pd

df=pd.read_csv("data/order_goods_app.txt",delimiter="\t",names=["order_id","goods_id","buy_count"])
# print(df.head(10))
print(df["buy_count"].value_counts())