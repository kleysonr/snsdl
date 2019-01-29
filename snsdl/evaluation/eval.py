from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
import os
import pandas as pd

# Credits - https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search

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
    def full_multiclass_report(y_true, y_pred, classes, png_output=None, show=True):
        """
        Multiclass or binary report.
        If binary (sigmoid output), set binary parameter to True
        """

        # Print accuracy score
        print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
        
        print("")
        
        # Print classification report
        print("Classification Report")
        Eval.classification_report(y_true,y_pred,digits=5)
        
        # Plot confusion matrix
        cnf_matrix = confusion_matrix(y_true,y_pred)
        print(cnf_matrix)

        Eval.plot_confusion_matrix(cnf_matrix,classes=classes, png_output=png_output, show=show)

    @staticmethod
    def classification_report(y_true, y_pred, digits=5, output_dir=None):

        report = classification_report(y_true,y_pred,digits=5)
        print(report)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            file = os.path.join(output_dir, 'classification_report.txt')

            f = open(file, 'w')
            f.write(report)
            f.close()

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, cmap=cm.Blues, png_output=None, show=True):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title='Normalized confusion matrix'
        else:
            title='Confusion matrix'

        f = plt.figure()

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if png_output is not None:
            os.makedirs(png_output, exist_ok=True)
            f.savefig(os.path.join(png_output,'confusion_matrix.png'), bbox_inches='tight')

        if show:
            plt.show()
            plt.close(f)
        else:
            plt.close(f)

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

        y_true_sample = np.array(y_true)[np.where(wrong_preds_indx == False)]
        y_pred_sample = np.array(y_pred)[np.where(wrong_preds_indx == False)]

        report = pd.DataFrame({'Image': wrong_predictions, 'True': y_true_sample, 'Predicted': y_pred_sample})

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

        f = plt.figure()

        # Create an axes instance
        ax = f.add_subplot(111)

        ## add patch_artist=True option to ax.boxplot() 
        ## to get fill color
        bp = ax.boxplot(data_to_plot, patch_artist=True)

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
        ax.set_xticklabels(classes)

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