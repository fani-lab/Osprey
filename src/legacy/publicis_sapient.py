# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # labels = ['GNB', 'LR', 'QDA', 'LDA', 'RF', 'KNN']
# # original = [0.61, 0.54, 0.70, 0.57, 0, 0]
# # smote = [0.58, 0.68, 0.84, 0.53, 0.91, 0.86]
# # width = 0.35
# # x = np.arange(len(labels))
# # fig, ax = plt.subplots()
# # rects1 = ax.bar(x - width/2, original, width, label='Original')
# # rects2 = ax.bar(x + width/2, smote, width, label='Smote')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('AUC')
# # ax.set_title('AUC Comparisons')
# # ax.set_xticks(x)
# # ax.set_xticklabels(labels)
# # ax.legend()
# #
# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)
# #
# # fig.tight_layout()
# #
# # plt.show()
#
#
# # import numpy as np
# # import matplotlib.pyplot as plt
# # # plt.figure(figsize=(28, 10), dpi=80)
# # labels = ['RF', 'RF-SMOTE']
# # accuracy = [0.954, 0.933]
# # specificity = [0.95, 0.94]
# # sensitivity = [0.14, 0.93]
# # PPV = [1, 0.93]
# # NPV = [0.0, 0.94]
# #
# #
# #
# # width = 0.13
# # x = np.arange(len(labels))
# # fig, ax = plt.subplots(figsize=(7.9,5))
# # rects1 = ax.bar(x, accuracy, width, label='Accuracy')
# # rects2 = ax.bar(x + width, specificity, width, label='Specificity')
# # rects3 = ax.bar(x + width*2, sensitivity, width, label='Sensitivity')
# # rects4 = ax.bar(x + width*3, PPV, width, label='PPV')
# # rects5 = ax.bar(x + width*4, NPV, width, label='NPV')
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Score')
# # ax.set_title('Random Forest Metric Comparisons')
# # ax.set_xticks(x)
# # ax.set_xticklabels(labels)
# # ax.legend()
# #
# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)
# # ax.bar_label(rects3, padding=3)
# # ax.bar_label(rects4, padding=3)
# # ax.bar_label(rects5, padding=3)
# #
# # fig.tight_layout()
# #
# # plt.show()
#
#
# # import time
# #
# # ascii_dict = {'A': 65, 'B': 66, 'C': 67, 'D': 68, 'E': 69, 'F': 70, 'G': 71, 'H': 72, 'I': 73, 'J': 74, 'K': 75,
# #               'L': 76, 'M': 77, 'N': 78, 'O': 79, 'P': 80, 'Q': 81, 'R': 82, 'S': 83, 'T': 84, 'U': 85, 'V': 86,
# #               'W': 87, 'X': 88, 'Y': 89, 'Z': 90}
# # start_time = time.time()
# #
# # # a = 'ZZYYZZZA'
# # # b = 'ZZYYZZZB'
# # # from variables import a,b
# # # ans = 'ZZYYZZYYZZZAZZZB'
# # str = ''
# # len_a = 0
# # len_b = 0
# #
# #
# # def change_and_return(str, a):
# #     str = str + a[0]
# #     a = a[1:]
# #     return str, a
# #
# #
# # def check_next_small(a, b):
# #     str = ''
# #     l = min(b, a, key=len)
# #     for i in range(len(l)):
# #         try:
# #             if ord(a[i]) < ord(b[i]):
# #                 return 'a'
# #             if ord(a[i]) > ord(b[i]):
# #                 return 'b'
# #         except:
# #             break
# #     if len(a) > len(b):
# #         return 'a'
# #     else:
# #         return 'b'
# #
# #
# # # len_a = len(a)
# # # len_b = len(b)
# #
# # def morgan_string(a, b):
# #     str = ''
# #     len_a = len(a)
# #     len_b = len(b)
# #     while len_a != 0 and len_b != 0:
# #         if ord(a[0]) < ord(b[0]):
# #             str, a = change_and_return(str, a)
# #             len_a = len_a - 1
# #         elif ord(a[0]) == ord(b[0]):
# #             if check_next_small(a, b) == 'a':
# #                 str, a = change_and_return(str, a)
# #                 len_a = len_a - 1
# #             elif check_next_small(a, b) == 'b':
# #                 str, b = change_and_return(str, b)
# #                 len_b = len_b - 1
# #             else:
# #                 str, b = change_and_return(str, b)
# #                 len_b = len_b - 1
# #         else:
# #             str, b = change_and_return(str, b)
# #             len_b = len_b - 1
# #     if len_a != 0:
# #         str = str + a
# #     if len_b != 0:
# #         str = str + b
# #     str = ''.join(str.split())
# #     # print(str)
# #
# #
# # from variables import a, b
# # morgan_string(a,b)
# # morgan_string(a,b)
# # morgan_string(a,b)
# # morgan_string(a,b)
# # morgan_string(a,b)
# # print("--- %s seconds ---" % (time.time() - start_time))
# # # asciiDict = {i+65: chr(i+65) for i in range(25)}
# # # asciiDict = dict((v,k) for k,v in asciiDict.items())
#
#
# # import math
# #
# #
# # def find_roots(a, b, c):
# #     return ((-b + math.sqrt(((b ** 2) - (4 * a * c)))) / (2 * a),
# #             (-b - math.sqrt(((b ** 2) - (4 * a * c)))) / (2 * a))
# #
# #
# # print(find_roots(2, 10, 8));
#
#
# import time
#
# # start_time = time.time()
# # generator = (i + 1 for i in range(9999999) if (i + 1) % 3 == 0 or (i + 1) % 5 == 0)
# # sum = 0
# # for number in generator:
# #     # print(number)
# #     sum = sum + number
# # print(sum)
# # print("--- %s seconds ---" % (time.time() - start_time))
# #
# # start_time = time.time()
# # s = 0
# # for i in range(9999999):
# #     if (i + 1) % 3 == 0 or (i + 1) % 5 == 0:
# #         # print(i+1)
# #         s = s + (i + 1)
# # print(s)
# # print("--- %s seconds ---" % (time.time() - start_time))
#
# import matplotlib.pyplot as plt
# from sklearn import datasets, metrics, model_selection, svm
#
# X, y = datasets.make_classification(random_state=0)
# X_train, X_test, y_train, y_test = model_selection.train_test_split(
#     X, y, random_state=0)
# clf = svm.SVC(random_state=0)
# clf.fit(X_train, y_train)
#
# metrics.plot_roc_curve(clf, X_test, y_test)
# plt.savefig(str(clf) + '.png')
# plt.show()
#
#
#
#
#
# print('No Skill: ROC AUC=%.3f' % (ns_auc))
# print('Logistic: ROC AUC=%.3f' % (lr_auc))
# # calculate roc curves
# ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
# lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# # plot the roc curve for the model
# plt.plot(fpr, tpr, linestyle='.', label=str(classifier))
# # axis labels
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # show the legend
# plt.legend()
# # show the plot
# plt.savefig(str(classifier) + '.png')
# plt.show()


