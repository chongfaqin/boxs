import pandas as pd

if __name__=="__main__":
    training_file="data/order_box_goods_app_filter_test.txt"
    data = pd.read_csv(training_file, delimiter="\t", names=["order_id", "feature_list", "label","warehouser"])
    print(data.head(10))

    feature_label = data[["feature_list","label"]].set_index('feature_list').T.to_dict('label')
    print(type(feature_label))

    index=0
    sub_feature_dict={}
    feature_list=data["feature_list"].unique()
    for nf in feature_list:
        index+=1
        feature_set=set(nf.split(","))
        if(len(feature_set)<2):
            print("skip",nf)
            continue
        sub_feature_dict[nf] = []
        for f in feature_list[index+1:]:
            other_feature_set=set(f.split(","))
            join_count=feature_set.issuperset(other_feature_set)
            if(join_count>0):
                # print(nf,f)
                sub_feature_dict[nf].append(f)

    # sub_feature_dict=sorted(sub_feature_dict)
    # print(sub_feature_dict)

    print("sub_feature_dict",len(sub_feature_dict))
    for k,f_list in sub_feature_dict.items():
        if(len(f_list)==0):
            continue
        k_lable=feature_label[k]
        for son_f in f_list:
            son_label=feature_label[son_f]
            if(int(son_label[0])>int(k_lable[0])):
                feature_label[son_f]=k_lable

    file = open("data/order_box_sub_goods_app_filter.txt", "w", encoding="utf-8")
    for items in data.values:
        k = items[1]
        if (k not in feature_label):
            continue
        file.write("%s\t%s\t%s\t%s\n" % (items[0], k, feature_label[k][0], items[3]))
    file.close()