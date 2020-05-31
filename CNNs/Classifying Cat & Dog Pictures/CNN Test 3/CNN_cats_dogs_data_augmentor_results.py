import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open('CNN_results_data_augmentor.pkl', 'rb') as f:
    results = pickle.load(f)

acc = pd.DataFrame(columns=['i', 'Epochs', 'Augmented', 'Acc', 'Loss', 'Train_Acc', 'Train_Loss'])

for row in results:
    row_vals = pd.DataFrame([row[0], row[1], row[2], row[3], row[4], row[5][-1], row[6][-1]]).T
    row_vals.columns = ['i', 'Epochs', 'Augmented', 'Acc', 'Loss', 'Train_Acc', 'Train_Loss']
    row_vals[['Acc', 'Loss', 'Train_Acc', 'Train_Loss']] = row_vals[['Acc', 'Loss', 'Train_Acc', 'Train_Loss']].astype(float)
    acc = acc.append(row_vals)

result_mean = acc.groupby(['Epochs', 'Augmented'],as_index=False).mean()
result_max = acc.groupby(['Epochs', 'Augmented'],as_index=False).max()
result_min = acc.groupby(['Epochs', 'Augmented'],as_index=False).min()
result_std = acc.groupby(['Epochs', 'Augmented']).std().reset_index()

Epochs_mean = acc.groupby(['Epochs'],as_index=False)['Acc', 'Loss', 'Train_Acc', 'Train_Loss'].mean()
Epochs_max = acc.groupby(['Epochs'],as_index=False)['Acc', 'Loss', 'Train_Acc', 'Train_Loss'].max()
Epochs_min = acc.groupby(['Epochs'],as_index=False)['Acc', 'Loss', 'Train_Acc', 'Train_Loss'].min()
Epochs_std = acc.groupby(['Epochs'])['Acc', 'Loss', 'Train_Acc', 'Train_Loss'].std().reset_index()

Augmented_mean = acc.groupby(['Augmented'],as_index=False)['Acc', 'Loss', 'Train_Acc', 'Train_Loss'].mean()
Augmented_max = acc.groupby(['Augmented'],as_index=False)['Acc', 'Loss', 'Train_Acc', 'Train_Loss'].max()
Augmented_min = acc.groupby(['Augmented'],as_index=False)['Acc', 'Loss', 'Train_Acc', 'Train_Loss'].min()
Augmented_std = acc.groupby(['Augmented'])['Acc', 'Loss', 'Train_Acc', 'Train_Loss'].std().reset_index()

def loss_acc_graphs(ind1, epochs, model_name):
    # x axis
    epochs = range(1,epochs+1) 
    #acc graph
    acc_r = results[ind1][5]
    Train_Acc_r = results[ind1][7]
    plt.plot(epochs, acc_r, 'bo', label='Training acc')
    plt.plot(epochs, Train_Acc_r, 'b', label='Validation acc')
    plt.title('Training and validation accuracy for best ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #loss graph
    loss_r = results[ind1][6]
    Train_Loss_r = results[ind1][8]
    plt.plot(epochs, loss_r, 'bo', label='Training loss')
    plt.plot(epochs, Train_Loss_r, 'b', label='Validation loss')
    plt.title('Training and validation loss for best ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return

def loss_acc_graphs_combined(ind1, epochs, model_name):
    # x axis
    epochs = range(1,epochs+1)
    f, axarr = plt.subplots(2, sharex=True)
    #acc graph
    acc_r = results[ind1][5]
    Train_Acc_r = results[ind1][7]
    axarr[0].plot(epochs, acc_r, 'bo', label='Training accuracy')
    axarr[0].plot(epochs, Train_Acc_r, 'b', label='Validation accuracy')
    axarr[0].set_title('Training and validation accuracy and loss for best ' + model_name)
    axarr[0].set_ylabel('Accuracy')
    axarr[0].legend()
    #loss graph
    loss_r = results[ind1][6]
    Train_Loss_r = results[ind1][8]
    axarr[1].plot(epochs, loss_r, 'bo', label='Training loss')
    axarr[1].plot(epochs, Train_Loss_r, 'b', label='Validation loss')
    axarr[1].set_xlabel('Epochs')
    axarr[1].set_ylabel('Loss')
    axarr[1].legend()
    return

loss_acc_graphs_combined(25, 30, 'augmented model')
loss_acc_graphs_combined(16, 20, 'non-augmented model')





