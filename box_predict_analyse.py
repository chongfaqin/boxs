import pandas as pd

df=pd.read_csv("data/test_boxs_predict.txt",encoding="utf-8",delimiter="\t",names=["order_id","predict","target"])
print(df.head(10))
order_sn=pd.read_csv("data/order_info.txt",encoding="utf-8",delimiter="\t",names=["order_id","order_sn"])
print(len(order_sn))
order_time=pd.read_csv("data/order_filter_label.txt",encoding="utf-8",delimiter="\t",names=["order_id","label","order_time","warehouse"])
# df=df[df["predict"]!=df["target"]]
df=df[["order_id","predict","target"]]
print(order_sn.head(10))
new_df=df.set_index('order_id').join(order_sn.set_index('order_id'),on="order_id",how="left")
new_df=new_df.join(order_time[["order_id","order_time"]].set_index('order_id'),on="order_id",how="left")
print(new_df.head(10))
new_df.to_csv("data/test_boxs_predict_result.csv")