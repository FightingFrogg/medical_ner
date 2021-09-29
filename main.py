# -*— coding: utf-8 -*-
# Name:main.py
# Author:SHAN
# Datetime:2021/9/29 上午9:36

import json


def main():
    data_file = './data/train.json'
    with open(data_file, 'r') as f:
        data = json.load(f)
        print(data[0])



if __name__ == '__main__':
    main()
