import functions as f
import sys
import os
import cv2
import argparse
import csv
from tqdm import tqdm
from classify import KanaClassifier
import classifyThree

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, help='path to trained model')
    ap.add_argument('-l', '--labelbin', required=True, help='path to label binarizer')
    ap.add_argument('-d', '--data', required=True, help='path to train data folder')
    ap.add_argument('-s', '--strategy', type=int, required=False, help='stratagey. 0-2', default=0)
    args = vars(ap.parse_args())
    
    kc = KanaClassifier(args['model'], args['labelbin'])
    strategies_str = ['strategy_max_each', 'strategy_max_product', 'strategy_max_sum']
    strategies = [classifyThree.strategy_max_each, classifyThree.strategy_max_product, classifyThree.strategy_max_sum]
    strategy = strategies[args['strategy']]
    strategy_str = strategies_str[args['strategy']]

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
    
    best_results = []
    for i in tqdm(range(100)):
        imgPath = os.path.join(args['data'], 'imgs', str(i) + '.jpg')
        img = cv2.imread(imgPath)
        filename = os.path.splitext(os.path.basename(imgPath))[0]
        # print(filename)
        results = classifyThree.classifyThree(img, filename, kc)
        best_result = strategy(results)
        best_results.append(best_result)

    with open(strategy_str + '.csv', mode='w', newline='') as output:
        output_writer = csv.writer(output, delimiter=',')

        output_writer.writerow(['ID', 'Unicode1', 'Unicode2', 'Unicode3'])
        false = 0
        for i in tqdm(range(100)):
            output_writer.writerow([i] + best_results[i])
        
            for j in range(3):
                if best_results[i][j] != answer[i][j]:
                    false += 1
                    break
        
        print('---- Incorrect rate: ', false/100)
                
if __name__ == "__main__":
    main()
