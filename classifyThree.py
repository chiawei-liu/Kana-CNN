import functions as f
import sys
import os
import cv2
import argparse
from classify import KanaClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, help='path to trained model')
    ap.add_argument('-l', '--labelbin', required=True, help='path to label binarizer')
    ap.add_argument('-i', '--image', required=True, help='path to input image')
    args = vars(ap.parse_args())
    imgPath = args['image']
    img = cv2.imread(imgPath)
    filename = os.path.splitext(os.path.basename(imgPath))[0]
    print(filename)
    kc = KanaClassifier(args['model'], args['labelbin'])
    classifyThree(img, filename, kc)

def classifyThree(img, filename, kc):
    segResults = [None for i in range(len(f.segMethods))]
    segResults = []
    for segMethod in f.segMethods:
        try:
            segResults.append(segMethod(filename, img))
            # print('succeed')
        except Exception as e:
            # print('fail')
            # print(e)
            segResults.append(None)
    # print(segResults)
    results = []
    for segResult in segResults:
        if segResult is not None:     
            result = []
            for img in segResult:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result.append(kc.classify(img)[0])
            results.append(result)
    return results

def strategy_max_sum(results):
    m_max = 0
    for result in results:
        m_sum = 0
        for kana in result:
            m_sum += kana['confidence']
        if m_sum > m_max:
            m_max = m_sum
            best_result = result

    return [best_result[0]['unicode'], best_result[1]['unicode'], best_result[2]['unicode']]

def strategy_max_product(results):
    m_max = 0
    for result in results:
        product = 1
        for kana in result:
            product *= kana['confidence']
        if product > m_max:
            m_max = product
            best_result = result 
    
    return [best_result[0]['unicode'], best_result[1]['unicode'], best_result[2]['unicode']]

def strategy_max_each(results):
    best_result = results[0]
    for result in results:
        for i in range(3):
            if result[i]['confidence'] > best_result[i]['confidence']:
                best_result[i] = result[i]
    

    return [best_result[0]['unicode'], best_result[1]['unicode'], best_result[2]['unicode']]
    '''
    try:
        segResults[i] = f.segmentation_1(filename, img)
        print('method 1')
    except:
        try:
            imgs = f.segmentation_2(filename, img)
            print('method 2')
        except:
            try:
                imgs = f.segmentation_3(filename, img)
                print('method 3')
            except:
                try:
                    imgs = f.segmentation_5(filename, img)
                    print('method 5')
                except:
                    imgs = None
                    print('all failed')
    '''    

if __name__ == "__main__":
    main()