import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

# Read in the Excel Spreadsheet with all the results of data augmentation experiments
results_all = pd.read_excel("AugExperiments.xlsx")

# Create a Dataframe with Only Rotation Augmentations and store metrics as lists
results_rot = results_all[results_all["Augmentation"]=="Rotate"]
results_rot = results_rot.dropna()
rot_deg = results_rot["Degree"].to_list()
rot_acc_val = results_rot["Ave_Acc_Val"].to_list()
rot_prec_val = results_rot["Ave_Prec_Val"].to_list()
rot_Rec_val = results_rot["Ave_Rec_Val"].to_list()
rot_f1_val = results_rot["Ave_F1_Val"].to_list()
rot_acc_test = results_rot["Ave_Acc_Test"].to_list()
rot_prec_test = results_rot["Ave_Prec_Test"].to_list()
rot_Rec_test = results_rot["Ave_Rec_Test"].to_list()
rot_f1_test = results_rot["Ave_F1_Test"].to_list()

rot_std_acc_test = results_rot["Std_Acc_Test"].to_list()
rot_std_prec_test = results_rot["Std_Prec_Test"].to_list()
rot_std_Rec_test = results_rot["Std_Rec_Test"].to_list()
rot_std_f1_test = results_rot["Std_F1_Test"].to_list()


# Create a Dataframe with Only Translation Augmentations and store metrics as lists
results_trans = results_all[results_all["Augmentation"]=="Translation"]
trans_deg = results_trans["Degree"].to_list()
trans_acc_val = results_trans["Ave_Acc_Val"].to_list()
trans_prec_val = results_trans["Ave_Prec_Val"].to_list()
trans_Rec_val = results_trans["Ave_Rec_Val"].to_list()
trans_f1_val = results_trans["Ave_F1_Val"].to_list()
trans_acc_test = results_trans["Ave_Acc_Test"].to_list()
trans_prec_test = results_trans["Ave_Prec_Test"].to_list()
trans_Rec_test = results_trans["Ave_Rec_Test"].to_list()
trans_f1_test = results_trans["Ave_F1_Test"].to_list()

trans_std_acc_test = results_trans["Std_Acc_Test"].to_list()
trans_std_prec_test = results_trans["Std_Prec_Test"].to_list()
trans_std_Rec_test = results_trans["Std_Rec_Test"].to_list()
trans_std_f1_test = results_trans["Std_F1_Test"].to_list()

# Axes labels
rot_labels = ['0','2','5','10','15','20','25','30' ]
trans_labels = ['0','2','4','6','8','10']
rot_x = np.arange(len(rot_labels))
trans_x = np.arange(len(trans_labels))
width = 0.4

# Configuration of subplots with all results

f3, f3axes = plt.subplots(2, 2)
# f3axes[0,0].bar(rot_x-(width/2),rot_acc_val, width/4, align = 'edge', label = "Mean Accuracy")
# f3axes[0,0].bar(rot_x-(width/4),rot_prec_val, width/4, align = 'edge', label = "Mean Precision")
# f3axes[0,0].bar(rot_x,rot_Rec_val, width/4, align = 'edge', label = "Mean Recall")
# f3axes[0,0].bar(rot_x+(width/4),rot_f1_val, width/4, align = 'edge', label = "Mean Accuracy")
# f3axes[0,0].grid()
# #f3axes[0,0].legend()
# f3axes[0,0].set_xticks(rot_x)
# f3axes[0,0].set_xticklabels(rot_labels)
# f3axes[0,0].set_ylabel("Peformance (%)")
# f3axes[0,0].set_title("Rotation")
# f3axes[0,0].set_ylim(50,108)


# f3axes[0,1].bar(trans_x-(width/2),trans_acc_val, width/4, align = 'edge', label = "Mean Accuracy")
# f3axes[0,1].bar(trans_x-(width/4),trans_prec_val, width/4, align = 'edge', label = "Mean Precision")
# f3axes[0,1].bar(trans_x,trans_Rec_val, width/4, align = 'edge', label = "Mean Recall")
# f3axes[0,1].bar(trans_x+(width/4),trans_f1_val, width/4, align = 'edge', label = "Mean F1-Score")
# f3axes[0,1].grid()
# f3axes[0,1].legend()
# f3axes[0,1].set_xticks(trans_x)
# f3axes[0,1].set_xticklabels(trans_labels)
# f3axes[0,1].set_ylabel("Peformance (%)")
# f3axes[0,1].set_title("Translation")
# f3axes[0,1].set_ylim(50,108)

f3axes[0,0].bar(rot_x-(width/2),rot_acc_test, width/4, align = 'edge', label = "Mean Accuracy")
f3axes[0,0].bar(rot_x-(width/4),rot_prec_test, width/4, align = 'edge', label = "Mean Precision")
f3axes[0,0].bar(rot_x,rot_Rec_test, width/4, align = 'edge', label = "Mean Recall")
f3axes[0,0].bar(rot_x+(width/4),rot_f1_test, width/4, align = 'edge', label = "Mean F1-Score")
f3axes[0,0].grid()
#f3axes[1,0].legend()
f3axes[0,0].set_xticks(rot_x)
f3axes[0,0].set_xticklabels(rot_labels)
f3axes[0,0].set_ylabel("Peformance (%)")
f3axes[0,0].set_title("Rotation")
f3axes[0,0].set_ylim(30,80)

# f3axes[1,1].plot(trans_deg,trans_acc_test, label = "Mean Accuracy")
# f3axes[1,1].plot(trans_deg,trans_prec_test, label = "Mean Precision")
# f3axes[1,1].plot(trans_deg,trans_Rec_test, label = "Mean Recall")
# f3axes[1,1].plot(trans_deg,trans_f1_test, label = "Mean F1-Score")
f3axes[0,1].bar(trans_x-(width/2),trans_acc_test, width/4, align = 'edge', label = "Mean Accuracy")
f3axes[0,1].bar(trans_x-(width/4),trans_prec_test, width/4, align = 'edge', label = "Mean Precision")
f3axes[0,1].bar(trans_x,trans_Rec_test, width/4, align = 'edge', label = "Mean Recall")
f3axes[0,1].bar(trans_x+(width/4),trans_f1_test, width/4, align = 'edge', label = "Mean F1-Score")
f3axes[0,1].grid()
#f3axes[1,1].legend()
f3axes[0,1].set_xticks(trans_x)
f3axes[0,1].set_xticklabels(trans_labels)
f3axes[0,1].set_ylabel("Peformance (%)")
f3axes[0,1].set_title("Translation")
f3axes[0,1].set_ylim(30,80)

f3axes[1,0].bar(rot_x-(width/2),rot_std_acc_test, width/4, align = 'edge', label = "Mean Accuracy")
f3axes[1,0].bar(rot_x-(width/4),rot_std_prec_test, width/4, align = 'edge', label = "Mean Precision")
f3axes[1,0].bar(rot_x,rot_std_Rec_test, width/4, align = 'edge', label = "Mean Recall")
f3axes[1,0].bar(rot_x+(width/4),rot_std_f1_test, width/4, align = 'edge', label = "Mean F1-Score")
f3axes[1,0].grid()
#f3axes[2,0].legend()
f3axes[1,0].set_xticks(rot_x)
f3axes[1,0].set_xticklabels(rot_labels)
f3axes[1,0].set_ylabel("Standard Deviation (%)")
f3axes[1,0].set_xlabel("Maximum Rotation ($^{\circ}$)")
f3axes[1,0].set_ylim(0,25)

# f3axes[2,1].plot(trans_deg,trans_std_acc_test, label = "Mean Accuracy")
# f3axes[2,1].plot(trans_deg,trans_std_prec_test, label = "Mean Precision")
# f3axes[2,1].plot(trans_deg,trans_std_Rec_test, label = "Mean Recall")
# f3axes[2,1].plot(trans_deg,trans_std_f1_test, label = "Mean F1-Score")
f3axes[1,1].bar(trans_x-(width/2),trans_std_acc_test, width/4, align = 'edge', label = "Mean Accuracy")
f3axes[1,1].bar(trans_x-(width/4),trans_std_prec_test, width/4, align = 'edge', label = "Mean Precision")
f3axes[1,1].bar(trans_x,trans_std_Rec_test, width/4, align = 'edge', label = "Mean Recall")
f3axes[1,1].bar(trans_x+(width/4),trans_std_f1_test, width/4, align = 'edge', label = "Mean F1-Score")
f3axes[1,1].grid()
#f3axes[2,1].legend()
f3axes[1,1].set_xticks(trans_x)
f3axes[1,1].set_xticklabels(trans_labels)
f3axes[1,1].set_ylabel("Standard Deviation (%)")
f3axes[1,1].set_xlabel("Maximum Translation (%)")
f3axes[1,1].set_ylim(0,25)

f3axes[1,1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2), fancybox=False, shadow=False, ncol=4)

plt.subplots_adjust(bottom=0.14, right=0.97, top=0.96, left=0.06, wspace=0.12, hspace=0.19)
plt.show(f3)



