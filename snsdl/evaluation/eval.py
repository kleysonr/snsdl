from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
import os.path

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
        print(classification_report(y_true,y_pred,digits=5))    
        
        # Plot confusion matrix
        cnf_matrix = confusion_matrix(y_true,y_pred)
        print(cnf_matrix)

        Eval.plot_confusion_matrix(cnf_matrix,classes=classes, png_output=png_output, show=show)

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