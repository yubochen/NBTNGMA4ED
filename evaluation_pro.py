import sys
def next_feature_judg(data,i):
    find_next = True
    k = 1
    while find_next:
        features = data[i].strip().split(" ")
        next_feature = data[i+ k].strip().split(" ")
        if next_feature == '':
            return 1,next_feature
        else:
            if next_feature[1][0] != "I" and next_feature[2][0] !="I":
                return 1,next_feature
            elif next_feature[1][0] == "I" and next_feature[2][0] == "I":
                if next_feature[1] == next_feature[2]:
                    k = k+1
                    return 1, next_feature
                else:
                    return 0, next_feature
            elif next_feature[1][0] == "I" or next_feature[2][0] == "I":
                if next_feature[1][0] == "I":
                    if next_feature[1][2] == "1":
                        return 1,next_feature
                    else:
                        return 0,next_feature
                elif next_feature[2][0] == "I":
                    if next_feature[2][2] == "1":
                        return 1,next_feature
                    else:
                        return 0,next_feature
                else:
                    return 0,next_feature

def write2file(result):
    with open("result.json", 'a', encoding="utf-8") as outfile:
        outfile.write(result)

def evaluation():
    with open("best_result/ner_predict.utf8",'r',encoding = "utf-8") as f:
        data = f.readlines()
    total_num = 0
    total_tri_num = 0
    tri_num  = 0
    find_num = 0
    corret_num = 0


    print (len(data))
    features_list = list()

    for i in range(len(data)-1):
        print (i)
        features = data[i].strip().split(" ")
        feature_1 = data[i+1].strip().split(" ")
        if features[0] == '':
            continue
        else:
            if features[1][0] == "B":
                total_num += 1
                if features[1][2] != "1":
                    total_tri_num += 1

                    if feature_1[1][0] == "I":
                        print(features)
                        print(feature_1)
                        tri_num += 1

            if features[2][0] == "B":
                if features[2][2] != "1":
                    # print (features)
                    find_num += 1

            if features[1][0] == "B" and features[2][0] == "B":
                if features[1][2] != "1" and features[2][2] != "1":
                    if features[1] == features[2]:
                        corret_num += 1
                        next_feature_judge, next_feature = next_feature_judg(data, i)
                        corret_num += next_feature_judge


    print (total_num)
    print(tri_num)
    print (total_tri_num)
    print (find_num)
    print (corret_num)
    r = corret_num/total_tri_num
    p = corret_num/find_num
    f = (p*r)/(p+r) * 2
    out = sys.stdout
    out.write('precision: %6.2f%%; ' % (100. * p))
    out.write('recall: %6.2f%%; ' % (100. * r))
    out.write('FB1: %6.2f\n' % (100. * f))
    result = 'precision: %6.2f%%' % (100. * p) + "\t" + ('recall: %6.2f%%' % (100. * r)) + "\t" + ('FB1: %6.2f\n' % (100. * f))
    write2file(result)

if __name__ == '__main__':
    evaluation()



