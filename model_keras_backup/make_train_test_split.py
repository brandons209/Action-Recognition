from random import shuffle
import csv
import glob
import numpy as np

action_classes = ['boxing', 'carrying', 'clapping', 'digging', 'jogging', 'openclosetrunk', 'running', 'throwing', 'walking', 'waving']
print(len(set(action_classes)))
def create_csvs():
    train = []
    test = []
    #total_train = []
    #total_test = []

    for myclass, directory in enumerate(action_classes):
        for filename in glob.glob('../data/aerial_clips/{}/*.avi'.format(directory)):
            group = np.random.randint(10)
            print(filename)
            if group < 3:
                test.append([filename, myclass, directory])
            else:
                train.append([filename, myclass, directory])

    shuffle(train)
    shuffle(test)
    # print('train', len(total_train))
    # print('test', len(total_test))

    with open('train.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'action'])
        mywriter.writerows(train)
        print('Training CSV file created successfully')

    with open('test.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'action'])
        mywriter.writerows(test)
        print('Testing CSV file created successfully')

    print('CSV files created successfully')


if __name__ == "__main__":
    create_csvs()
