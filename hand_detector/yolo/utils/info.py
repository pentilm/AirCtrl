import os


def data_info(type='train'):
    dataset_directory = '../../../EgoGesture Dataset/'
    folder_names = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive', 'SingleSix', 'SingleSeven',
                    'SingleEight']

    if type is 'train':
        nTrain = 0
        for folder in folder_names:
            nTrain = nTrain + len(os.listdir(dataset_directory + folder + '/'))
        return nTrain

    elif type is 'valid':
        nValid = 0
        for folder in folder_names:
            nValid = nValid + len(os.listdir(dataset_directory + folder + '/'))
        return nValid
