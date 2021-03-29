
if __name__=='__main__':

    order_goods_dict={}
    with open("data/order_goods_app.txt","r",encoding="utf-8") as file:
        for line in file.readlines():
            line_arr=line.strip().split("\t")
            # print(line_arr)
            if(line_arr[0] in order_goods_dict):
                order_goods_dict[line_arr[0]].add(line_arr[1]+":"+line_arr[2])
            else:
                order_goods_dict[line_arr[0]]=set([line_arr[1]+":"+line_arr[2]])

    print(len(order_goods_dict))

    order_id=7062800
    order_id_set=order_goods_dict[str(order_id)]

    result_order_set=set()
    for k,v in order_goods_dict.items():
        # print(k,v)
        join_count=len(order_id_set & v)
        br=order_id_set.issuperset(v)
        if(join_count>=2 and br):
            print(k,v)
            result_order_set.add(k)

    print(result_order_set)
