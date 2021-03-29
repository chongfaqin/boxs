import pandas as pd

if __name__=="__main__":
    training_file="data/order_box_goods_app.txt"
    data = pd.read_csv(training_file, delimiter="\t", names=["order_id", "feature_list", "label","warehouser"])
    print(data.head(10))
    data_groupby=data.groupby(["feature_list", "label"]).count()
    print(data_groupby.head(10))
    data_feature_groupby=data.groupby(["feature_list"]).count()
    print(data_feature_groupby.head(10))
    # for k in data_groupby.index:
    #     print(k,data_groupby[k])
    feature_label={}
    for idx,vals in zip(data_groupby.index,data_groupby.values):
        # print(idx,vals)
        # feature_label_count[idx]=vals[0]
        fea=idx[0]
        label=idx[1]
        value=vals[0]
        if(fea in feature_label):
            feature_label[fea][label]=value
        else:
            feature_label[fea]={label:value}

    feature_label_only = {}
    for k, v in feature_label.items():
        sv = sorted(v.items(), key=lambda item: item[1], reverse=True)
        feature_label_only[k] = sv[0][0]

    file=open("data/order_box_goods_app_filter.txt","w",encoding="utf-8")
    for k in data["feature_list"].unique():
        # k = items[1]
        if (k not in feature_label_only):
            continue
        file.write("%s\t%s\t%s\t%s\n" % (str(0),k,feature_label_only[k],str(0)))
    file.close()