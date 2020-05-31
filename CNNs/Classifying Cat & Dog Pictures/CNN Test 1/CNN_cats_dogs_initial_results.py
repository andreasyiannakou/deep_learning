import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open('CNN_results_initial.pkl', 'rb') as f:
    results = pickle.load(f)

acc = pd.DataFrame(columns=['Activation', 'Padding', 'Pooling', 'Kernel', 'Acc', 'Loss'])

for row in results:
    row_vals = pd.DataFrame([row[1], row[2], row[3], row[4], row[5], row[6]]).T
    row_vals.columns = ['Activation', 'Padding', 'Pooling', 'Kernel', 'Acc', 'Loss']
    row_vals[['Loss', 'Acc']] = row_vals[['Loss', 'Acc']].astype(float)
    acc = acc.append(row_vals)

result_mean = acc.groupby(['Activation', 'Padding', 'Pooling', 'Kernel'],as_index=False).mean()
result_max = acc.groupby(['Activation', 'Padding', 'Pooling', 'Kernel'],as_index=False).max()
result_min = acc.groupby(['Activation', 'Padding', 'Pooling', 'Kernel'],as_index=False).min()
#result_std = acc.groupby(['Activation', 'Padding', 'Pooling', 'Kernel'],as_index=False).std()

Activation_mean = acc.groupby(['Activation'],as_index=False)['Acc', 'Loss'].mean()
Activation_max = acc.groupby(['Activation'],as_index=False)['Acc', 'Loss'].max()
Activation_min = acc.groupby(['Activation'],as_index=False)['Acc', 'Loss'].min()
Activation_std = acc.groupby(['Activation'])['Acc','Loss'].std().reset_index()

Padding_mean = acc.groupby(['Padding'],as_index=False)['Acc', 'Loss'].mean()
Padding_max = acc.groupby(['Padding'],as_index=False)['Acc', 'Loss'].max()
Padding_min = acc.groupby(['Padding'],as_index=False)['Acc', 'Loss'].min()
Padding_std = acc.groupby(['Padding'])['Acc','Loss'].std().reset_index()

Pooling_mean = acc.groupby(['Pooling'],as_index=False)['Acc', 'Loss'].mean()
Pooling_max = acc.groupby(['Pooling'],as_index=False)['Acc', 'Loss'].max()
Pooling_min = acc.groupby(['Pooling'],as_index=False)['Acc', 'Loss'].min()
Pooling_std = acc.groupby(['Pooling'])['Acc','Loss'].std().reset_index()

Kernel_mean = acc.groupby(['Kernel'],as_index=False)['Acc', 'Loss'].mean()
Kernel_max = acc.groupby(['Kernel'],as_index=False)['Acc', 'Loss'].max()
Kernel_min = acc.groupby(['Kernel'],as_index=False)['Acc', 'Loss'].min()
Kernel_std = acc.groupby(['Kernel'])['Acc','Loss'].std().reset_index()

def loss_acc_graphs(ind1, epochs, model_name):
    # x axis
    epochs = range(1,epochs+1) 
    #acc graph
    acc_r = results[ind1][7]
    val_acc_r = results[ind1][9]
    plt.plot(epochs, acc_r, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_r, 'b', label='Validation acc')
    plt.title('Training and validation accuracy for best ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #loss graph
    loss_r = results[ind1][8]
    val_loss_r = results[ind1][10]
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
    acc_r = results[ind1][7]
    val_acc_r = results[ind1][9]
    axarr[0].plot(epochs, acc_r, 'bo', label='Training accuracy')
    axarr[0].plot(epochs, val_acc_r, 'b', label='Validation accuracy')
    axarr[0].set_title('Training and validation accuracy and loss for best ' + model_name)
    axarr[0].set_ylabel('Accuracy')
    axarr[0].legend()
    #loss graph
    loss_r = results[ind1][8]
    val_loss_r = results[ind1][10]
    axarr[1].plot(epochs, loss_r, 'bo', label='Training loss')
    axarr[1].plot(epochs, val_loss_r, 'b', label='Validation loss')
    axarr[1].set_xlabel('Epochs')
    axarr[1].set_ylabel('Loss')
    axarr[1].legend()
    return

loss_acc_graphs_combined(36, 10, 'ReLU model')
loss_acc_graphs_combined(25, 10, 'SReLU model')





