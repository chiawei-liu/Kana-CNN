import os
import csv
from tqdm import tqdm
import argparse
from apyori import apriori

def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-d', '--data', required=True, help='path to test data folder')
    args = vars(ap.parse_args())
    answer = []
    with open(os.path.join(args['data'], 'annotations.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count = 1
                continue
            else:
                answer.append(row[1:])

    # answer = answer[:1000]
    #print(answer)
    #print(len(answer))
    '''
    answer=[
        [1,2,3],
        [1,2,4],
        [0,2,5],
        [0,2,3]
    ]
    '''
    f = open('rules.txt', 'w')
    association_rules = apriori(answer, min_support=0.00001, min_confidence=0.0013, max_length=3)
    association_results = list(association_rules)
    # print(association_results)
    # print(association_rules[0])
    for item in association_results:
        #print(item)   
    # first index of the inner list
    # Contains base item and add item
        pair = item[0] 
        items = [x for x in pair]
        if len(items) < 2:
            continue
        f.write(str(item))
        '''
        print("Rule: " + items[0] + " -> " + items[1])

        #second index of the inner list
        print("Support: " + str(item[1]))

        #third index of the list located at 0th
        #of the third index of the inner list

        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        '''

        # print("=====================================")
        f.write("\n=====================================\n")

                
if __name__ == "__main__":
    main()