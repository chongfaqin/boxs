import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

def get_goods():
    # goods_id_dict={}
    # 打开数据库连接
    db = MySQLdb.connect(host='wonderfulloffline.rds.inagora.org',port=3306,user='wonderfull_ai',password='868wxRHrPaTKkjvC', db='wonderfull_ai_online', charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select goods_id FROM goods where sell_count>0")
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    goods_id_dict={}
    for line in data:
        print(line)
        key="g_" + str(line[0])
        if(key in goods_id_dict):
            continue
        goods_id_dict[key] = 1
    db.close()
    return goods_id_dict

def get_goods_code():
    goods_code={}
    with open("data/box_goods_code.txt","r") as file:
        for line in file.readlines():
            line_arr=line.strip().split(",")
            goods_code[line_arr[0]]=line_arr[1]
    return goods_code

if __name__ == '__main__':

    goods_code=get_goods_code()
    goods_id_dict=get_goods()
    print("goods_list:",len(goods_id_dict))

    idx=len(goods_code)+1
    for k,v in goods_id_dict.items():
        if (k not in goods_code):
            goods_code[k] = idx
            idx = idx + 1

    file_writer=open("data/goods_code.txt","w",encoding="utf-8")
    for k,v in goods_code.items():
        file_writer.write(k+","+str(v)+"\n")
    file_writer.close()