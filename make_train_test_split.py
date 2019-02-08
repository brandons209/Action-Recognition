from random import shuffle
import csv
import glob
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-video_path", type=str, default="data/", help="path to root folder containing videos seperated into folders by class, unless for transfer then it should be the path to the folder containing videos")
parser.add_argument("-transfer", type=int, default=0, help="1 if creating csv files for youtube8m for weight transfer, default 0.")
parser.add_argument("-valid_split", type=float, default=0.3, help="decimal percentage of how much of data to split for validation")
options = parser.parse_args()

#['boxing', 'carrying', 'clapping', 'digging', 'jogging', 'openclosetrunk', 'running', 'throwing', 'walking', 'waving']
if options.transfer == 0:
    global folder_paths
    folder_paths = glob.glob("{}*".format(options.video_path))
    for folder in folder_paths:
        if not os.path.isdir(folder):
            folder_paths.remove(folder)
    global action_classes
    action_classes = [folder.split('/')[-1] for folder in folder_paths]
    print("Number of classes: {}".format(len(action_classes)))
else:
    global video_paths
    video_paths = glob.glob("{}*".format(options.video_path))

def create_csvs():
    train = []
    test = []

    if options.transfer == 0:
        for myclass, directory in enumerate(folder_paths):
            for filename in glob.glob('{}/*'.format(directory)):
                group = np.random.randint(100) + 1

                if group < options.valid_split * 100:
                    test.append([filename, myclass, directory.split("/")[-1]])
                else:
                    train.append([filename, myclass, directory.split("/")[-1]])
        shuffle(train)
        shuffle(test)

        with open('data/train.csv', 'w') as csvfile:
            mywriter = csv.writer(csvfile)
            mywriter.writerow(['path', 'class', 'action'])
            mywriter.writerows(train)
            print('Training CSV file created successfully at data/train.csv')

        with open('data/test.csv', 'w') as csvfile:
            mywriter = csv.writer(csvfile)
            mywriter.writerow(['path', 'class', 'action'])
            mywriter.writerows(test)
            print('Testing CSV file created successfully at data/test.csv')
    else:
        for directory in video_paths:
            group = np.random.randint(100) + 1

            if group < options.valid_split * 100:
                test.append([directory])
            else:
                train.append([directory])
        shuffle(train)
        shuffle(test)

        with open('data/train_wt.csv', 'w') as csvfile:
            mywriter = csv.writer(csvfile)
            mywriter.writerow(['path'])
            mywriter.writerows(train)
            print('Training_wt CSV file created successfully at data/train_wt.csv')

        with open('data/test_wt.csv', 'w') as csvfile:
            mywriter = csv.writer(csvfile)
            mywriter.writerow(['path'])
            mywriter.writerows(test)
            print('Testing_wt CSV file created successfully at data/test_wt.csv')

    print('CSV files created successfully')


if __name__ == "__main__":
    create_csvs()
