import json
import datetime
order_box_index={"Z0000":0,"Z0001":1,"Z0002":2,"Z0003":3,"Z0004":4,"Z0005":5,"Z0006":6,
                 "Z0007":7,"Z0008":8,"Z0009":9,"Z0010":10,"Z0011":11,"Z0012":12,"Z0013":13,"Z0014":14,"Z0015":15,
                 "W0000":0,"W0001":1,"W0002":2,"W0003":3,"W0004":4,"W0005":5,"W0006":6,
                 "W0007":7,"W0008":8,"W0009":9,"W0010":10,"W0011":11,"W0012":12,"W0013":13,"W0014":14,"W0015":15}

if __name__=="__main__":
    result = []
    boxs = {}
    dateftime = datetime.datetime.strptime("1/1/2020 00:00:00", "%d/%m/%Y %H:%M:%S")
    with open("data/order_action_app.txt", "r", encoding="utf-8") as file:
        for line in file.readlines():
            # print(line)
            arr = line.strip().split("\t")
            # print(arr)
            try:
                if (len(arr) == 6):
                    strftime = datetime.datetime.strptime(arr[4], "%d/%m/%Y %H:%M:%S")
                    if(strftime<dateftime):
                        print(arr[4])
                        continue

                    data = json.loads(arr[5])
                    if ("box" in data):
                        # print(arr[1],data["box"])
                        if (len(data["box"]) != 5):
                            continue
                        # if ("郑州保税仓" in arr[2] and order_box_index[data["box"]]>6):
                        #     print(arr[2],order_box_index[data["box"]])
                        #     continue
                        if ("郑州保税仓" in arr[2]):
                            print(arr[2])
                            continue
                        if (data["box"].startswith("W") or data["box"].startswith("Z")):
                            result.append([arr[1], order_box_index[data["box"]],arr[4],arr[2]])
                            print(arr[1], data["box"])
                            if (data["box"] in boxs):
                                boxs[data["box"]] += 1
                            else:
                                boxs[data["box"]] = 1
            except Exception as e:
                print(e, line.strip())

    print("result:", len(result))
    print("boxs:", boxs)

    file=open("data/order_filter_label.txt","w",encoding="utf-8")
    for item in result:
        file.write("%s\t%d\t%s\t%s\n"%(item[0],item[1],item[2],item[3]))
    file.close()