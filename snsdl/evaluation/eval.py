from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pycm import *

# Partial Credits - https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search

class Eval:

    @staticmethod
    def plot_history(history, png_output=None, show=True):

        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
        
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
        
        # As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
        
        # Two subplots, unpack the axes array immediately
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(15,6))

        ## Loss
        for l in loss_list:
            ax1.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            ax1.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        
        ax1.set_title('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ## Accuracy
        for l in acc_list:
            ax2.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            ax2.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Fine-tune figure; make subplots farther from each other.
        f.subplots_adjust(hspace=0.5)

        if png_output is not None:
            os.makedirs(png_output, exist_ok=True)
            f.savefig(os.path.join(png_output,'loss_acc.png'), bbox_inches='tight')

        if show:
            plt.show()
            plt.close(f)
        else:
            plt.close(f)

    @staticmethod
    def full_multiclass_report(y_true, y_pred, output_dir=None):
        """
        Multiclass report.
        """

        # Print accuracy score
        print('Accuracy : ' + str(accuracy_score(y_true,y_pred)) + '\n')
        
        report = classification_report(y_true, y_pred, digits=5)
        print(report)

        if output_dir is not None:

            os.makedirs(output_dir, exist_ok=True)

            file = os.path.join(output_dir, 'classification_report.txt')

            f = open(file, 'w')
            f.write(report)
            f.close()

    @staticmethod
    def confusion_matrix_report(y_true, y_pred, output_dir=None, largeCM=False, overall_param=None, class_param=None, class_name=None, matrix_save=True, normalize=False, color='silver'):

        cm = ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)
        file = os.path.join(output_dir, 'cm_report')

        # If CM matrix is large, save report to disk
        if largeCM:
            if output_dir is None:
                print('[WARN] For a large confusion matrix, `output_dir` must be specified.')

            else:
                cm.save_csv(file, class_param=class_param, class_name=class_name, matrix_save=matrix_save, normalize=normalize)

                if os.path.exists(os.path.join(output_dir, 'cm_report.csv')):

                    csv_f = pd.read_csv(os.path.join(output_dir, 'cm_report.csv'))

                    with open(os.path.join(output_dir, 'cm_report.txt'), 'w') as f:
                        f.write(csv_f.to_string())

                    col_names = list(csv_f.columns)[1:]

                    if os.path.exists(os.path.join(output_dir, 'cm_report_matrix.csv')):

                        csv_f = pd.read_csv(os.path.join(output_dir, 'cm_report_matrix.csv'), names=col_names)
                        csv_f.insert(loc=0, column='Class', value=col_names)

                        with open(os.path.join(output_dir, 'cm_report_matrix.txt'), 'w') as f:
                            f.write(csv_f.to_string())

                        #
                        results = []
                        for i in csv_f.index:

                            row = csv_f.loc[[i]]
                            a = row.loc[:, (row != 0).any()]
                            class_name = a.iloc[0,0] 
                            a = a.rename(columns = {class_name:'*'+class_name+'*'})

                            results.append(a.to_string())

                        with open(os.path.join(output_dir, 'cm_report_matrix_summary.txt'), 'w') as f:
                            for r in results: 
                                f.write(r) 
                                f.write('\r\n\r\n') 

        else:
            print(cm)

            if output_dir is not None:
                cm.save_html(file, overall_param=overall_param, class_param=class_param, class_name=class_name, color=color)

    @staticmethod
    def plot_wrong_predictions(samples, y_true, y_pred, classes, size=9):

        wrong_preds_indx = (y_true == y_pred)
        wrong_predictions = samples[np.where(wrong_preds_indx == False)][:size]

        y_true_sample = y_true[np.where(wrong_preds_indx == False)][:size]
        y_pred_sample = y_pred[np.where(wrong_preds_indx == False)][:size]

        # Create figure with 3x3 sub-plots.
        f, axes = plt.subplots(3, 3)
        f.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):

            if wrong_predictions[i].shape[2] == 1:
                shape = wrong_predictions[i].shape[:2]
                cmap = 'binary'
            else:
                shape = wrong_predictions[i].shape
                cmap = 'bgr'

            # Plot image.
            ax.imshow(wrong_predictions[i].reshape(shape), cmap=cmap)

            # Show true and predicted classes.
            xlabel = "True: {0}, Pred: {1}".format(y_true_sample[i], y_pred_sample[i])

            ax.set_xlabel(xlabel)
            
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
            
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

        plt.close(f)

    @staticmethod
    def wrong_predictions_report(samples, y_true, y_pred, output_dir):

        wrong_preds_indx = (np.array(y_true) == np.array(y_pred))
        wrong_predictions = np.array(samples)[np.where(wrong_preds_indx == False)]

        for i, s in enumerate(wrong_predictions):
            tokens = s.split(os.path.sep)
            wrong_predictions[i] = '/'.join(tokens[-2:])

        y_true_sample = np.array(y_true)[np.where(wrong_preds_indx == False)]
        y_pred_sample = np.array(y_pred)[np.where(wrong_preds_indx == False)]

        report = pd.DataFrame({'Sample': wrong_predictions, 'True': y_true_sample, 'Predicted': y_pred_sample})

        os.makedirs(output_dir, exist_ok=True)

        file = os.path.join(output_dir, 'wrong_predictions_report.txt')

        f = open(file, 'w')
        f.write(report.to_string())
        f.close()

    @staticmethod
    def boxplot_report(samples, y_true, y_pred, probs, classes, boxplot_output=None, report_output=None, show=True):

        correct_preds_indx = (np.array(y_true) == np.array(y_pred))
        correct_predictions = np.where(correct_preds_indx == True)[0]

        data_to_plot = []

        for c in classes:
            indx_by_class = np.where(np.array(y_pred) == c)[0]
            correct_indx_by_class = np.intersect1d(indx_by_class, correct_predictions)
            class_probs = probs[correct_indx_by_class] * 100
            data_to_plot.append(class_probs)

            if report_output is not None:
                os.makedirs(report_output, exist_ok=True)

                names_probs = np.array(samples)[correct_indx_by_class]
                report = pd.DataFrame({'Image': names_probs, 'Probability': class_probs})
                report = report.sort_values(by='Probability', ascending=False)
                file = os.path.join(report_output, 'probs_class_{}.txt'.format(c))

                fi = open(file, 'w')
                fi.write(report.to_string())
                fi.close()

        # Calculate chart area size
        leftmargin = 0.5 # inches
        rightmargin = 0.5 # inches
        categorysize = 0.5 # inches
        figwidth = leftmargin + rightmargin + ((len(classes)+1) * categorysize)

        # Create figure
        f = plt.figure(figsize=(figwidth, 5))

        # Create an axes instance and ajust the subplot size
        ax = f.add_subplot(111)
        f.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)

        ## add patch_artist=True option to ax.boxplot() 
        ## to get fill color
        bp = ax.boxplot(data_to_plot, patch_artist=True, positions=np.arange(len(classes)))

        ## change outline color, fill color and linewidth of the boxes
        for box in bp['boxes']:
            # change outline color
            box.set( color='#7570b3', linewidth=2)
            # change fill color
            box.set( facecolor = '#1b9e77' )

        ## change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#b2df8a', linewidth=2)

        ## change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.5)

        ## Custom x-axis labels
        ax.set_xticklabels(classes, rotation=45, ha='right')

        ## Remove top axes and right axes ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()        

        if boxplot_output is not None:
            os.makedirs(boxplot_output, exist_ok=True)
            f.savefig(os.path.join(boxplot_output,'box_plot.png'), bbox_inches='tight')

        if show:
            plt.show()
            plt.close(f)
        else:
            plt.close(f)
