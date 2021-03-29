import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

def get_category():
    user_category = {}
    goods_category={}
    # 打开数据库连接
    db = MySQLdb.connect(host='wonderfulloffline.rds.inagora.org',port=3306,user='wonderfull_ai',password='868wxRHrPaTKkjvC', db='wonderfull_ai_online', charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select cat_id,cat_name,level FROM category where level=3")
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    for line in data:
        print(line)
        goods_category["c_" + str(line[0])] = line[1]
    db.close()
    return goods_category

def get_brand():
    user_category = {}
    goods_brand={}
    # 打开数据库连接
    db = MySQLdb.connect(host='wonderfulloffline.rds.inagora.org',port=3306,user='wonderfull_ai',password='868wxRHrPaTKkjvC', db='wonderfull_ai_online', charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select brand_id,brand_name FROM brand")
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    for line in data:
        print(line)
        goods_brand["b_" + str(line[0])] = line[1]
    db.close()
    return goods_brand

def get_brand_category():
    category_brand_dict={}
    # 打开数据库连接
    db = MySQLdb.connect(host='wonderfulloffline.rds.inagora.org',port=3306,user='wonderfull_ai',password='868wxRHrPaTKkjvC', db='wonderfull_ai_online', charset='utf8')
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # 使用execute方法执行SQL语句
    cursor.execute("select cat_id,brand_id FROM goods where sell_count>0")
    # 使用 fetchone() 方法获取一条数据
    data = cursor.fetchall()
    for line in data:
        print(line)
        category_brand_dict["j_" + str(line[0])+"_"+str(line[1])] = 1
    db.close()
    return category_brand_dict

if __name__ == '__main__':
    category=get_category()
    print("category:",len(category))
    # category_brand=get_brand_category()
    # print("category_brand:",len(category_brand))
    brand=get_brand()
    print("brand:",len(brand))

    code={}
    idx=1
    for k,v in category.items():
        if (k not in code):
            code[k] = idx
            idx = idx + 1

    for k,v in brand.items():
        if (k not in code):
            code[k] = idx
            idx = idx + 1

    file_writer=open("data/box_feature_code.txt","w",encoding="utf-8")
    for k,v in code.items():
        file_writer.write(k+","+str(v)+"\n")
    file_writer.close()