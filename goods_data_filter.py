import json
import datetime
import re
order_box_index={"Z0000":0,"Z0001":1,"Z0002":2,"Z0003":3,"Z0004":4,"Z0005":5,"Z0006":6,
                 "Z0007":7,"Z0008":8,"Z0009":9,"Z0010":10,"Z0011":11,"Z0012":12,"Z0013":13,"Z0014":14,"Z0015":15,
                 "W0000":0,"W0001":1,"W0002":2,"W0003":3,"W0004":4,"W0005":5,"W0006":6,
                 "W0007":7,"W0008":8,"W0009":9,"W0010":10,"W0011":11,"W0012":12,"W0013":13,"W0014":14,"W0015":15}

if __name__=="__main__":
    result = []
    boxs = {}
    dateftime = datetime.datetime.strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    with open("data/order_action_app.txt", "r", encoding="utf-8") as file:
        for line in file.readlines():
            # print(line)
            arr = line.strip().split("\t")
            # print(arr)
            try:
                if (len(arr) == 4):
                    strftime = datetime.datetime.strftime(arr[3], "%Y-%m-%d %H:%M:%S")
                    if(strftime<dateftime):
                        print(arr[3])
                        continue

                    # data = json.loads(arr[2])
                    # if ("box" in data):
                    # print(arr[1],data["box"])
                    # if ("郑州保税仓" in arr[1]):
                    #     print(arr[1])
                    #     continue
                    match=re.search("[WZ]00[0-2][0-9]",arr[2])
                    if (not match):
                        # print(arr[2])
                        continue
                    box_type=match.group(0)
                    print(arr[0], box_type)
                    if("郑州保税仓" in arr[1] and order_box_index[box_type]>6):
                        print("filter",box_type)
                        continue

                    if (box_type.startswith("W") or box_type.startswith("Z")):
                        result.append([arr[0],order_box_index[box_type],arr[3],arr[1]])
                        if (box_type in boxs):
                            boxs[box_type] += 1
                        else:
                            boxs[box_type] = 1
            except Exception as e:
                print(e, line.strip())

    print("result:", len(result))
    print("boxs:", boxs)

    file=open("data/order_filter_label.txt","w",encoding="utf-8")
    for item in result:
        file.write("%s\t%d\t%s\t%s\n"%(item[0],item[1],item[2],item[3]))
    file.close()