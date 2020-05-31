import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open('CNN_results_improvements.pkl', 'rb') as f:
    results = pickle.load(f)

acc = pd.DataFrame(columns=['Neurons', 'Dropout', 'BatchNorm', 'Acc', 'Loss'])

for row in results:
    row_vals = pd.DataFrame([row[0], row[1], row[2], row[3], row[4]]).T
    row_vals.columns = ['Neurons', 'Dropout', 'BatchNorm', 'Acc', 'Loss']
    row_vals[['Loss', 'Acc']] = row_vals[['Loss', 'Acc']].astype(float)
    acc = acc.append(row_vals)

result_mean = acc.groupby(['Neurons', 'Dropout', 'BatchNorm'],as_index=False).mean()
result_max = acc.groupby(['Neurons', 'Dropout', 'BatchNorm'],as_index=False).max()
result_min = acc.groupby(['Neurons', 'Dropout', 'BatchNorm'],as_index=False).min()
result_std = acc.groupby(['Neurons', 'Dropout', 'BatchNorm']).std().reset_index()

Neurons_mean = acc.groupby(['Neurons'],as_index=False)['Acc', 'Loss'].mean()
Neurons_max = acc.groupby(['Neurons'],as_index=False)['Acc', 'Loss'].max()
Neurons_min = acc.groupby(['Neurons'],as_index=False)['Acc', 'Loss'].min()
Neurons_std = acc.groupby(['Neurons'])['Acc','Loss'].std().reset_index()

Dropout_mean = acc.groupby(['Dropout'],as_index=False)['Acc', 'Loss'].mean()
Dropout_max = acc.groupby(['Dropout'],as_index=False)['Acc', 'Loss'].max()
Dropout_min = acc.groupby(['Dropout'],as_index=False)['Acc', 'Loss'].min()
Dropout_std = acc.groupby(['Dropout'])['Acc','Loss'].std().reset_index()

BatchNorm_mean = acc.groupby(['BatchNorm'],as_index=False)['Acc', 'Loss'].mean()
BatchNorm_max = acc.groupby(['BatchNorm'],as_index=False)['Acc', 'Loss'].max()
BatchNorm_min = acc.groupby(['BatchNorm'],as_index=False)['Acc', 'Loss'].min()
BatchNorm_std = acc.groupby(['BatchNorm'])['Acc','Loss'].std().reset_index()

def loss_acc_graphs(ind1, epochs, model_name):
    # x axis
    epochs = range(1,epochs+1) 
    #acc graph
    acc_r = results[ind1][5]
    val_acc_r = results[ind1][7]
    plt.plot(epochs, acc_r, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_r, 'b', label='Validation acc')
    plt.title('Training and validation accuracy for best ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #loss graph
    loss_r = results[ind1][6]
    val_loss_r = results[ind1][8]
    plt.plot(epochs, loss_r, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_r, 'b', label='Validation loss')
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
    val_acc_r = results[ind1][7]
    axarr[0].plot(epochs, acc_r, 'bo', label='Training accuracy')
    axarr[0].plot(epochs, val_acc_r, 'b', label='Validation accuracy')
    axarr[0].set_title('Training and validation accuracy and loss for best ' + model_name)
    axarr[0].set_ylabel('Accuracy')
    axarr[0].legend()
    #loss graph
    loss_r = results[ind1][6]
    val_loss_r = results[ind1][8]
    axarr[1].plot(epochs, loss_r, 'bo', label='Training loss')
    axarr[1].plot(epochs, val_loss_r, 'b', label='Validation loss')
    axarr[1].set_xlabel('Epochs')
    axarr[1].set_ylabel('Loss')
    axarr[1].legend()
    return

loss_acc_graphs_combined(4, 10, 'accuracy model')
loss_acc_graphs_combined(6, 10, 'loss model')





