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

# def get_goods_indexs(code):
#     goods_index={}
#     # 打开数据库连接
#     db = MySQLdb.connect(host='wonderfulloffline.rds.inagora.org',port=3306,user='wonderfull_ai',password='868wxRHrPaTKkjvC', db='wonderfull_ai_online', charset='utf8')
#     # 使用cursor()方法获取操作游标
#     cursor = db.cursor()
#     # 使用execute方法执行SQL语句
#     cursor.execute("select goods_id,cat_id,brand_id FROM goods where sell_count>0")
#     # 使用 fetchone() 方法获取一条数据
#     data = cursor.fetchall()
#     for line in data:
#         goods_key="g_"+str(line[0])
#         # indexs = []
#         if(goods_key in code):
#             # indexs.append(code[goods_key])
#             goods_index[str(line[0])] = code[goods_key]
#         else:
#             # indexs.append(0)
#             goods_index[str(line[0])] = 0
#     db.close()
#     return goods_index

def get_code_index():
    code={}
    with open("data/goods_code.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            arr=line.strip().split(",")
            code[arr[0]]=arr[1]
    return code

def get_order_goods():
    order_buy_goods_dic={}
    order_goods_list_dict={}
    # max=2
    with open("data/order_goods_app.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            arr=line.strip().split("\t")
            if (int(arr[2]) > 30):
                continue
            if(arr[0] in order_goods_list_dict):
                order_goods_list_dict[arr[0]].append(arr[1])
            else:
                order_goods_list_dict[arr[0]]=[arr[1]]
            order_buy_goods_dic[arr[0] + ":" + arr[1]] = arr[2]
    return order_goods_list_dict,order_buy_goods_dic

def get_scala_max_min(v,max):
    # divisor=np.log(v+1)-np.log(min)
    # dividend=np.log(max)-np.log(min)
    return np.round(v/max,4)

def get_goods_dict():
    goods_min_label={}
    with open("data/goods_min_box.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            line_arr=line.strip().split("\t")
            goods_min_label[line_arr[0]]=line_arr[1]
    return goods_min_label

if __name__=='__main__':

    df=pd.read_csv("data/order_filter_label.txt",delimiter="\t",names=["order_id","label","order_time","warehouse"])
    print(df["label"].value_counts())

    code=get_code_index()
    # goods_index=get_goods_indexs(code)
    order_goods_dict,goods_buy_dict=get_order_goods()
    print("code:",len(code))
    # print("goods_index:",len(goods_index))
    print("order_goods_dict:",len(order_goods_dict))
    print("goods_buy_dict:",len(goods_buy_dict))

    goods_min_label=get_goods_dict()
    print("goods_min_label",len(goods_min_label))

    no_goods_order_count=0
    has_goods_order_count=0
    order_box_file=open("data/order_box_goods_app.txt","w",encoding="utf-8")
    # for items in df[["order_sn","label"]].values:
    update_count=0
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
        box_lable = items[1]
        for goods_id in goods_id_list:
            goods_id_key="g_"+goods_id
            if(goods_id_key not in code):
                flag=True
                print("goods_id of no feature index:",goods_id)
                # goods_feature_index = 0
                # feature[goods_feature_index] = 1
                continue
            else:
                goods_feature_index=int(code[goods_id_key])
                goods_buy = goods_buy_dict[order_id + ":" + goods_id]
                if (goods_feature_index in feature):
                    feature[goods_feature_index] += int(goods_buy)
                else:
                    feature[goods_feature_index] = int(goods_buy)

            '''
            修正标签
            '''
            if(goods_id in goods_min_label):
                new_label=goods_min_label[goods_id]
                if(int(box_lable)<int(new_label)):
                    print("update",box_lable,new_label)
                    box_lable=new_label
                    update_count+=1

        if(len(feature)==0 or flag):
            print(flag,order_id)
            continue
        try:
            for k in sorted(feature):
                feature_list.append(str(k)+":"+str(get_scala_max_min(feature[k],30)))
        except Exception as e:
            print(e,items,k,goods_id_list)
            continue
        has_goods_order_count+=1
        feature_str=",".join(feature_list)
        whouse=items[3]
        order_box_file.write("%s\t%s\t%s\t%s\n" %(order_id,feature_str,box_lable,whouse))
    order_box_file.close()
    print("no_goods_order_count:",no_goods_order_count)
    print("has_goods_order_count:",has_goods_order_count)
    print("update_lable_count:",update_count)
