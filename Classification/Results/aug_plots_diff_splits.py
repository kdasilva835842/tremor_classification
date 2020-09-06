import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

# Read in the Excel Spreadsheet with all the results of data augmentation experiments
results_all_80_20 = pd.read_excel("AugExperiments.xlsx")
results_all_70_30 = pd.read_excel("AugExperiments_70_30.xlsx")

# Create a Dataframe with Only Rotation Augmentations for 80 20 split and store metrics as lists
results_rot_80_20 = results_all_80_20[results_all_80_20["Augmentation"]=="Rotate"]
results_rot_80_20 = results_rot_80_20.dropna()
rot_deg_80_20 = results_rot_80_20["Degree"].to_list()
rot_acc_val_80_20 = results_rot_80_20["Ave_Acc_Val"].to_list()
rot_prec_val_80_20 = results_rot_80_20["Ave_Prec_Val"].to_list()
rot_Rec_val_80_20 = results_rot_80_20["Ave_Rec_Val"].to_list()
rot_f1_val_80_20 = results_rot_80_20["Ave_F1_Val"].to_list()
rot_acc_test_80_20 = results_rot_80_20["Ave_Acc_Test"].to_list()
rot_prec_test_80_20 = results_rot_80_20["Ave_Prec_Test"].to_list()
rot_Rec_test_80_20 = results_rot_80_20["Ave_Rec_Test"].to_list()
rot_f1_test_80_20 = results_rot_80_20["Ave_F1_Test"].to_list()

rot_std_acc_test_80_20 = results_rot_80_20["Std_Acc_Test"].to_list()
rot_std_prec_test_80_20 = results_rot_80_20["Std_Prec_Test"].to_list()
rot_std_Rec_test_80_20 = results_rot_80_20["Std_Rec_Test"].to_list()
rot_std_f1_test_80_20 = results_rot_80_20["Std_F1_Test"].to_list()

# Create a Dataframe with Only Rotation Augmentations for 70 230 split and store metrics as lists
results_rot_70_30 = results_all_70_30[results_all_70_30["Augmentation"]=="Rotate"]
results_rot_70_30 = results_rot_70_30.dropna()
rot_deg_70_30 = results_rot_70_30["Degree"].to_list()
rot_acc_val_70_30 = results_rot_70_30["Ave_Acc_Val"].to_list()
rot_prec_val_70_30 = results_rot_70_30["Ave_Prec_Val"].to_list()
rot_Rec_val_70_30 = results_rot_70_30["Ave_Rec_Val"].to_list()
rot_f1_val_70_30 = results_rot_70_30["Ave_F1_Val"].to_list()
rot_acc_test_70_30 = results_rot_70_30["Ave_Acc_Test"].to_list()
rot_prec_test_70_30 = results_rot_70_30["Ave_Prec_Test"].to_list()
rot_Rec_test_70_30 = results_rot_70_30["Ave_Rec_Test"].to_list()
rot_f1_test_70_30 = results_rot_70_30["Ave_F1_Test"].to_list()

rot_std_acc_test_70_30 = results_rot_70_30["Std_Acc_Test"].to_list()
rot_std_prec_test_70_30 = results_rot_70_30["Std_Prec_Test"].to_list()
rot_std_Rec_test_70_30 = results_rot_70_30["Std_Rec_Test"].to_list()
rot_std_f1_test_70_30 = results_rot_70_30["Std_F1_Test"].to_list()

# Create a Dataframe with Only Translation Augmentations for 80 20 splits and store metrics as lists
results_trans_80_20 = results_all_80_20[results_all_80_20["Augmentation"]=="Translation"]
trans_deg_80_20 = results_trans_80_20["Degree"].to_list()
trans_acc_val_80_20 = results_trans_80_20["Ave_Acc_Val"].to_list()
trans_prec_val_80_20 = results_trans_80_20["Ave_Prec_Val"].to_list()
trans_Rec_val_80_20 = results_trans_80_20["Ave_Rec_Val"].to_list()
trans_f1_val_80_20 = results_trans_80_20["Ave_F1_Val"].to_list()
trans_acc_test_80_20 = results_trans_80_20["Ave_Acc_Test"].to_list()
trans_prec_test_80_20 = results_trans_80_20["Ave_Prec_Test"].to_list()
trans_Rec_test_80_20 = results_trans_80_20["Ave_Rec_Test"].to_list()
trans_f1_test_80_20 = results_trans_80_20["Ave_F1_Test"].to_list()

trans_std_acc_test_80_20 = results_trans_80_20["Std_Acc_Test"].to_list()
trans_std_prec_test_80_20 = results_trans_80_20["Std_Prec_Test"].to_list()
trans_std_Rec_test_80_20 = results_trans_80_20["Std_Rec_Test"].to_list()
trans_std_f1_test_80_20 = results_trans_80_20["Std_F1_Test"].to_list()

# Create a Dataframe with Only Translation Augmentations for 70 30 splits and store metrics as lists
results_trans_70_30 = results_all_70_30[results_all_70_30["Augmentation"]=="Translation"]
trans_deg_70_30 = results_trans_70_30["Degree"].to_list()
trans_acc_val_70_30 = results_trans_70_30["Ave_Acc_Val"].to_list()
trans_prec_val_70_30 = results_trans_70_30["Ave_Prec_Val"].to_list()
trans_Rec_val_70_30 = results_trans_70_30["Ave_Rec_Val"].to_list()
trans_f1_val_70_30 = results_trans_70_30["Ave_F1_Val"].to_list()
trans_acc_test_70_30 = results_trans_70_30["Ave_Acc_Test"].to_list()
trans_prec_test_70_30 = results_trans_70_30["Ave_Prec_Test"].to_list()
trans_Rec_test_70_30 = results_trans_70_30["Ave_Rec_Test"].to_list()
trans_f1_test_70_30 = results_trans_70_30["Ave_F1_Test"].to_list()

trans_std_acc_test_70_30 = results_trans_70_30["Std_Acc_Test"].to_list()
trans_std_prec_test_70_30 = results_trans_70_30["Std_Prec_Test"].to_list()
trans_std_Rec_test_70_30 = results_trans_70_30["Std_Rec_Test"].to_list()
trans_std_f1_test_70_30 = results_trans_70_30["Std_F1_Test"].to_list()

# Axes labels
rot_labels = ['0','2','5','10','15','20','25','30' ]
trans_labels = ['0','2','4','6','8','10']
rot_x = np.arange(len(rot_labels))
trans_x = np.arange(len(trans_labels))
width = 0.4

# Configuration of subplots with all rotation results for different splits

f1, f1axes = plt.subplots(2, 2)
# f1axes[0,0].bar(rot_x-(width/2),rot_acc_val_80_20, width/4, align = 'edge', label = "Mean Accuracy")
# f1axes[0,0].bar(rot_x-(width/4),rot_prec_val_80_20, width/4, align = 'edge', label = "Mean Precision")
# f1axes[0,0].bar(rot_x,rot_Rec_val_80_20, width/4, align = 'edge', label = "Mean Recall")
# f1axes[0,0].bar(rot_x+(width/4),rot_f1_val_80_20, width/4, align = 'edge', label = "Mean Accuracy")
# f1axes[0,0].grid()
# #f1axes[0,0].legend()
# f1axes[0,0].set_xticks(rot_x)
# f1axes[0,0].set_xticklabels(rot_labels)
# f1axes[0,0].set_ylabel("Peformance (%)")
# f1axes[0,0].set_title("Rotation (80-20 split)")
# f1axes[0,0].set_ylim(50,108)

# f1axes[0,1].bar(rot_x-(width/2),rot_acc_val_70_30, width/4, align = 'edge', label = "Mean Accuracy")
# f1axes[0,1].bar(rot_x-(width/4),rot_prec_val_70_30, width/4, align = 'edge', label = "Mean Precision")
# f1axes[0,1].bar(rot_x,rot_Rec_val_70_30, width/4, align = 'edge', label = "Mean Recall")
# f1axes[0,1].bar(rot_x+(width/4),rot_f1_val_70_30, width/4, align = 'edge', label = "Mean Accuracy")
# f1axes[0,1].grid()
# #f1axes[0,0].legend()
# f1axes[0,1].set_xticks(rot_x)
# f1axes[0,1].set_xticklabels(rot_labels)
# f1axes[0,1].set_ylabel("Peformance (%)")
# f1axes[0,1].set_title("Rotation (70-30 split)")
# f1axes[0,1].set_ylim(50,108)

f1axes[0,0].bar(rot_x-(width/2),rot_acc_test_80_20, width/4, align = 'edge', label = "Mean Accuracy")
f1axes[0,0].bar(rot_x-(width/4),rot_prec_test_80_20, width/4, align = 'edge', label = "Mean Precision")
f1axes[0,0].bar(rot_x,rot_Rec_test_80_20, width/4, align = 'edge', label = "Mean Recall")
f1axes[0,0].bar(rot_x+(width/4),rot_f1_test_80_20, width/4, align = 'edge', label = "Mean F1-Score")
f1axes[0,0].grid()
#f1axes[1,0].legend()
f1axes[0,0].set_xticks(rot_x)
f1axes[0,0].set_xticklabels(rot_labels)
f1axes[0,0].set_ylabel("Peformance (%)")
f1axes[0,0].set_title("Rotation (80-20 split)")
f1axes[0,0].set_ylim(40,80)

f1axes[0,1].bar(rot_x-(width/2),rot_acc_test_70_30, width/4, align = 'edge', label = "Mean Accuracy")
f1axes[0,1].bar(rot_x-(width/4),rot_prec_test_70_30, width/4, align = 'edge', label = "Mean Precision")
f1axes[0,1].bar(rot_x,rot_Rec_test_70_30, width/4, align = 'edge', label = "Mean Recall")
f1axes[0,1].bar(rot_x+(width/4),rot_f1_test_70_30, width/4, align = 'edge', label = "Mean F1-Score")
f1axes[0,1].grid()
#f1axes[1,1].legend()
f1axes[0,1].set_xticks(rot_x)
f1axes[0,1].set_xticklabels(rot_labels)
f1axes[0,1].set_ylabel("Peformance (%)")
f1axes[0,1].set_title("Rotation (70-30 split)")
f1axes[0,1].set_ylim(40,80)

f1axes[1,0].bar(rot_x-(width/2),rot_std_acc_test_80_20, width/4, align = 'edge', label = "Mean Accuracy")
f1axes[1,0].bar(rot_x-(width/4),rot_std_prec_test_80_20, width/4, align = 'edge', label = "Mean Precision")
f1axes[1,0].bar(rot_x,rot_std_Rec_test_80_20, width/4, align = 'edge', label = "Mean Recall")
f1axes[1,0].bar(rot_x+(width/4),rot_std_f1_test_80_20, width/4, align = 'edge', label = "Mean F1-Score")
f1axes[1,0].grid()
#f1axes[2,0].legend()
f1axes[1,0].set_xticks(rot_x)
f1axes[1,0].set_xticklabels(rot_labels)
f1axes[1,0].set_ylabel("Standard Deviation (%)")
f1axes[1,0].set_xlabel("Maximum Rotation ($^{\circ}$)")
f1axes[1,0].set_ylim(0,20)

f1axes[1,1].bar(rot_x-(width/2),rot_std_acc_test_70_30, width/4, align = 'edge', label = "Mean Accuracy")
f1axes[1,1].bar(rot_x-(width/4),rot_std_prec_test_70_30, width/4, align = 'edge', label = "Mean Precision")
f1axes[1,1].bar(rot_x,rot_std_Rec_test_70_30, width/4, align = 'edge', label = "Mean Recall")
f1axes[1,1].bar(rot_x+(width/4),rot_std_f1_test_70_30, width/4, align = 'edge', label = "Mean F1-Score")
f1axes[1,1].grid()
#f1axes[2,1].legend()
f1axes[1,1].set_xticks(rot_x)
f1axes[1,1].set_xticklabels(rot_labels)
f1axes[1,1].set_ylabel("Standard Deviation (%)")
f1axes[1,1].set_xlabel("Maximum Rotation ($^{\circ}$)")
f1axes[1,1].set_ylim(0,20)

f1axes[1,1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2), fancybox=False, shadow=False, ncol=4)

plt.subplots_adjust(bottom=0.14, right=0.97, top=0.96, left=0.06, wspace=0.12, hspace=0.19)
plt.show(f1)

# ******************************************************************************

# Configuration of subplots with all translation results for different splits

f2, f2axes = plt.subplots(2, 2)
# f2axes[0,0].bar(trans_x-(width/2),trans_acc_val_80_20, width/4, align = 'edge', label = "Mean Accuracy")
# f2axes[0,0].bar(trans_x-(width/4),trans_prec_val_80_20, width/4, align = 'edge', label = "Mean Precision")
# f2axes[0,0].bar(trans_x,trans_Rec_val_80_20, width/4, align = 'edge', label = "Mean Recall")
# f2axes[0,0].bar(trans_x+(width/4),trans_f1_val_80_20, width/4, align = 'edge', label = "Mean Accuracy")
# f2axes[0,0].grid()
# #f2axes[0,0].legend()
# f2axes[0,0].set_xticks(trans_x)
# f2axes[0,0].set_xticklabels(trans_labels)
# f2axes[0,0].set_ylabel("Peformance (%)")
# f2axes[0,0].set_title("Translation (80-20 split)")
# f2axes[0,0].set_ylim(50,108)

# f2axes[0,1].bar(trans_x-(width/2),trans_acc_val_70_30, width/4, align = 'edge', label = "Mean Accuracy")
# f2axes[0,1].bar(trans_x-(width/4),trans_prec_val_70_30, width/4, align = 'edge', label = "Mean Precision")
# f2axes[0,1].bar(trans_x,trans_Rec_val_70_30, width/4, align = 'edge', label = "Mean Recall")
# f2axes[0,1].bar(trans_x+(width/4),trans_f1_val_70_30, width/4, align = 'edge', label = "Mean Accuracy")
# f2axes[0,1].grid()
# #f2axes[0,1].legend()
# f2axes[0,1].set_xticks(trans_x)
# f2axes[0,1].set_xticklabels(trans_labels)
# f2axes[0,1].set_ylabel("Peformance (%)")
# f2axes[0,1].set_title("Translation (70-30 split)")
# f2axes[0,1].set_ylim(50,108)

f2axes[0,0].bar(trans_x-(width/2),trans_acc_test_80_20, width/4, align = 'edge', label = "Mean Accuracy")
f2axes[0,0].bar(trans_x-(width/4),trans_prec_test_80_20, width/4, align = 'edge', label = "Mean Precision")
f2axes[0,0].bar(trans_x,trans_Rec_test_80_20, width/4, align = 'edge', label = "Mean Recall")
f2axes[0,0].bar(trans_x+(width/4),trans_f1_test_80_20, width/4, align = 'edge', label = "Mean F1-Score")
f2axes[0,0].grid()
#f2axes[1,0].legend()
f2axes[0,0].set_xticks(trans_x)
f2axes[0,0].set_xticklabels(trans_labels)
f2axes[0,0].set_ylabel("Peformance (%)")
f2axes[0,0].set_title("Translation (80-20 split)")
f2axes[0,0].set_ylim(30,80)

f2axes[0,1].bar(trans_x-(width/2),trans_acc_test_70_30, width/4, align = 'edge', label = "Mean Accuracy")
f2axes[0,1].bar(trans_x-(width/4),trans_prec_test_70_30, width/4, align = 'edge', label = "Mean Precision")
f2axes[0,1].bar(trans_x,trans_Rec_test_70_30, width/4, align = 'edge', label = "Mean Recall")
f2axes[0,1].bar(trans_x+(width/4),trans_f1_test_70_30, width/4, align = 'edge', label = "Mean F1-Score")
f2axes[0,1].grid()
#f2axes[1,1].legend()
f2axes[0,1].set_xticks(trans_x)
f2axes[0,1].set_xticklabels(trans_labels)
f2axes[0,1].set_ylabel("Peformance (%)")
f2axes[0,1].set_title("Translation (70-30 split)")
f2axes[0,1].set_ylim(30,80)

f2axes[1,0].bar(trans_x-(width/2),trans_std_acc_test_80_20, width/4, align = 'edge', label = "Mean Accuracy")
f2axes[1,0].bar(trans_x-(width/4),trans_std_prec_test_80_20, width/4, align = 'edge', label = "Mean Precision")
f2axes[1,0].bar(trans_x,trans_std_Rec_test_80_20, width/4, align = 'edge', label = "Mean Recall")
f2axes[1,0].bar(trans_x+(width/4),trans_std_f1_test_80_20, width/4, align = 'edge', label = "Mean F1-Score")
f2axes[1,0].grid()
#f2axes[2,0].legend()
f2axes[1,0].set_xticks(trans_x)
f2axes[1,0].set_xticklabels(trans_labels)
f2axes[1,0].set_ylabel("Standard Deviation (%)")
f2axes[1,0].set_xlabel("Maximum Translation ($^{\circ}$)")
f2axes[1,0].set_ylim(0,25)

f2axes[1,1].bar(trans_x-(width/2),trans_std_acc_test_70_30, width/4, align = 'edge', label = "Mean Accuracy")
f2axes[1,1].bar(trans_x-(width/4),trans_std_prec_test_70_30, width/4, align = 'edge', label = "Mean Precision")
f2axes[1,1].bar(trans_x,trans_std_Rec_test_70_30, width/4, align = 'edge', label = "Mean Recall")
f2axes[1,1].bar(trans_x+(width/4),trans_std_f1_test_70_30, width/4, align = 'edge', label = "Mean F1-Score")
f2axes[1,1].grid()
#f2axes[2,1].legend()
f2axes[1,1].set_xticks(trans_x)
f2axes[1,1].set_xticklabels(trans_labels)
f2axes[1,1].set_ylabel("Standard Deviation (%)")
f2axes[1,1].set_xlabel("Maximum Translation ($^{\circ}$)")
f2axes[1,1].set_ylim(0,25)


f2axes[1,1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2), fancybox=False, shadow=False, ncol=4)

plt.subplots_adjust(bottom=0.14, right=0.97, top=0.96, left=0.06, wspace=0.12, hspace=0.19)
plt.show(f2)

