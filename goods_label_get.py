import pandas as pd
import MySQLdb

# 打开数据库连接
db = MySQLdb.connect(host='wonderfulloffline.rds.wandougongzhu.cn',port=3306,user='wonderfull',password='TdYey2FAJyxNNaf8', db='wonderfull_online', charset='utf8')
def get_goods_id(jan_code):
    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select goods_id,skuid FROM goods_rel_skuid where skuid="+str(jan_code))
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    # goods_data = {}
    for line in data:
        # goods_data[line[1]] = line[0]
        return line[0]

if __name__=="__main__":
    data=pd.read_csv("data/boxs_label.csv")
    print(data.head(10))
    data=data[["JAN","最小箱型预估","最多可装商品数预估"]]
    data=data.dropna()
    print(data.head(10))
    with open("data/goods_min_box.txt","w",encoding="utf-8") as file:
        for items in data.values:
            goods_id=get_goods_id(items[0])
            if(items[1]=='NaN' or items[1]==None):
                continue
            print(goods_id,int(items[1]))
            file.write("%s\t%d\n" %(str(goods_id),int(items[1])))