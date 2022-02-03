import numpy as np
from preprocess.finder import class_finder

""" Ground Truth Outputs """
gt_probability = np.load('../data/gt/gt_prob_classes5.npy')
gt_position = np.load('../data/gt/gt_pos_classes5.npy')

""" Predicted Outputs """
pr_probability = np.load('../data/classes5/pr_prob_classes5.npy')
pr_position = np.load('../data/classes5/pr_pos_classes5.npy')


def error_finder(prob, gt_pos, pos):
    squared_diff = np.square(gt_pos - pos)
    error = 0
    for i in range(0, 5):
        if prob[i] == 1:
            error = error + np.sqrt(squared_diff[2 * i] + squared_diff[2 * i + 1])
    error = error / sum(prob)
    return error


class_number = [0, 1, 2, 3, 4]
accuracy = [[], [], [], [], []]
precision = [[], [], [], [], []]
recall = [[], [], [], [], []]
f1_score = [[], [], [], [], []]
fingertip_err = np.array([0, 0, 0, 0, 0])

threshold_values = [0.5]
for threshold in threshold_values:
    print('Threshold: ', threshold)

    true_positive = [0, 0, 0, 0, 0]
    false_positive = [0, 0, 0, 0, 0]
    false_negative = [0, 0, 0, 0, 0]
    true_negative = [0, 0, 0, 0, 0]

    for id in class_number:
        for gt_prob, pr_prob, gt_pos, pr_pos in zip(gt_probability, pr_probability, gt_position, pr_position):
            prob = np.asarray([(p >= threshold) * 1.0 for p in pr_prob])
            gt_class = class_finder(gt_prob)
            pr_class = class_finder(prob)
            error = error_finder(prob, gt_pos, pr_pos)

            if gt_class == pr_class == id and error < 15:
                true_positive[id] = true_positive[id] + 1
                fingertip_err[id] = fingertip_err[id] + error
            elif gt_class == pr_class == id and error >= 15:
                """ pixel error greater than 15 is considered as wrong 
                    prediction 
                """
                false_negative[id] = false_negative[id] + 1
            else:
                if gt_class == id:
                    false_negative[id] = false_negative[id] + 1
                elif pr_class == id:
                    false_positive[id] = false_positive[id] + 1
                else:
                    true_negative[id] = true_negative[id] + 1

    print('True positive:', true_positive)
    print('False positive:', false_positive)
    print('False Negative:', false_negative)
    print('True Negative:', true_negative)

    for k in range(0, len(class_number)):
        try:
            p = true_positive[k] / (true_positive[k] + false_positive[k])
        except ZeroDivisionError:
            p = 0
        try:
            r = true_positive[k] / (true_positive[k] + false_negative[k])
        except ZeroDivisionError:
            r = 0
        precision[k].append(round(p, 6))
        recall[k].append(round(r, 6))
        f1 = 2 * ((p * r) / (p + r))
        f1_score[k].append(round(f1, 6))

    true_positive = np.array(true_positive)
    true_negative = np.array(true_negative)
    false_positive = np.array(false_positive)
    false_negative = np.array(false_negative)

    a = (true_positive + true_negative)
    b = (true_positive + true_negative + false_positive + false_negative)
    accuracy = a / b
    accuracy = accuracy * 100
    accuracy = np.round(accuracy, 4)
    fingertip_err = np.round(fingertip_err / true_positive, 2)

print('')
print('Accuracy =', accuracy)
print('precision =', precision, np.mean(precision))
print('recall =', recall, np.mean(recall))
print('F1 Score  =', f1_score)

print('')
print('Fingertip Error =', fingertip_err)
print('Mean Error =', np.round(np.mean(fingertip_err), 2))
