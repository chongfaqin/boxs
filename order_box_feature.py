import json
import pandas as pd
import numpy as np
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

order_box_index={"Z0000":0,"Z0001":1,"Z0002":2,"Z0003":3,"Z0004":4,"Z0005":5,"Z0006":6,
                 "Z0007":7,"Z0008":8,"Z0009":9,"Z0010":10,"Z0011":11,"Z0012":12,"Z0013":13,"Z0014":14,"Z0015":15,
                 "W0000":0,"W0001":1,"W0002":2,"W0003":3,"W0004":4,"W0005":5,"W0006":6,
                 "W0007":7,"W0008":8,"W0009":9,"W0010":10,"W0011":11,"W0012":12,"W0013":13,"W0014":14,"W0015":15}

def get_goods_indexs(code):
    goods_index={}
    # 打开数据库连接
    db = MySQLdb.connect(host='wonderfulloffline.rds.inagora.org',port=3306,user='wonderfull_ai',password='868wxRHrPaTKkjvC', db='wonderfull_ai_online', charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select goods_id,cat_id,brand_id FROM goods where sell_count>0")
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    for line in data:
        category_key="c_"+str(line[1])
        category_brand_key="b_"+str(line[2])

        if(category_key not in code or category_brand_key not in code):
            continue
        goods_index[str(line[0])] = [code[category_key],code[category_brand_key]]
    db.close()
    return goods_index

def get_code_index():
    code={}
    with open("data/box_feature_code.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            arr=line.strip().split(",")
            code[arr[0]]=arr[1]
    return code

def get_order_goods():
    order_goods_dic={}
    order={}
    with open("data/order_goods_app.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            arr=line.strip().split("\t")
            if (int(arr[2]) > 30):
                continue
            if(arr[0] in order):
                order[arr[0]].append(arr[1])
            else:
                order[arr[0]]=[arr[1]]
            order_goods_dic[arr[0]+":"+arr[1]] = arr[2]
    return order,order_goods_dic

def get_scala_max_min(v,max):
    return np.round(v/max,4)

if __name__=='__main__':

    df = pd.read_csv("data/order_label.txt", delimiter="\t", names=["order_id", "label"])
    print(df["label"].value_counts())

    code=get_code_index()
    goods_index=get_goods_indexs(code)
    order_goods_dict,goods_buy_dict=get_order_goods()
    print("code:",len(code))
    print("goods_index:",len(goods_index))
    print("order_goods_dict:",len(order_goods_dict))
    print("goods_buy_dict:",len(goods_buy_dict))

    no_goods_order_count=0
    has_goods_order_count=0
    order_box_file=open("data/order_box_feature.txt","w",encoding="utf-8")
    # for items in df[["order_sn","label"]].values:
    for items in df.values:
        # print(items,type(items[0]))
        order_id=str(items[0])
        if(order_id not in order_goods_dict):
            print("order_id of no goods:",order_id)
            no_goods_order_count+=1
            continue
        # print("order_id has goods:",order_id)
        goods_id_list=order_goods_dict[order_id]
        feature_list = []
        feature = {}
        flag=False
        for goods_id in goods_id_list:
            if(goods_id not in goods_index):
                flag=True
                print("goods_id of no feature index:",goods_id)
                continue
            goods_feature_index=goods_index[goods_id]
            goods_buy = goods_buy_dict[order_id + ":" + goods_id]
            for idx in goods_feature_index:
                if(idx in feature):
                    feature[idx]+=int(goods_buy)
                else:
                    feature[idx]=int(goods_buy)
        if(len(feature)==0 or flag):
            print(flag,order_id)
            continue
        for k in sorted(feature):
            try:
                feature_list.append(str(k)+":"+str(get_scala_max_min(feature[k],30)))
            except Exception as e:
                print(items,k,goods_id_list)
        has_goods_order_count+=1
        feature_str=",".join(feature_list)
        box_lable=items[1]
        order_box_file.write("%s\t%s\t%d\n" %(order_id,feature_str,box_lable))
    order_box_file.close()
    print("no_goods_order_count:",no_goods_order_count)
    print("has_goods_order_count:",has_goods_order_count)
