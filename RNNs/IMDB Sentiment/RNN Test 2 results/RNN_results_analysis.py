import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open('RNN_results_2.pkl', 'rb') as f:
    results = pickle.load(f)

acc = pd.DataFrame(columns=['Name', 'Acc', 'Loss'])

for row in results:
    row_vals = pd.DataFrame([row[1], row[2], row[3]]).T
    row_vals.columns = ['Name', 'Acc', 'Loss']
    row_vals[['Loss', 'Acc']] = row_vals[['Loss', 'Acc']].astype(float)
    acc = acc.append(row_vals)

result_mean = acc.groupby('Name').mean()
result_std = acc.groupby('Name').std()
result_max = acc.groupby('Name').max()
result_min = acc.groupby('Name').min()

#Best ReLU = 0, best SReLU = 64


def loss_acc_graphs(ind1, epochs, model_name):
    # x axis
    epochs = range(1,epochs+1) 
    #acc graph
    acc_r = results[ind1][4]
    val_acc_r = results[ind1][6]
    plt.plot(epochs, acc_r, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_r, 'b', label='Validation acc')
    plt.title('Training and validation accuracy for best ' + model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #loss graph
    loss_r = results[ind1][5]
    val_loss_r = results[ind1][7]
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
    acc_r = results[ind1][4]
    val_acc_r = results[ind1][6]
    axarr[0].plot(epochs, acc_r, 'bo', label='Training accuracy')
    axarr[0].plot(epochs, val_acc_r, 'b', label='Validation accuracy')
    axarr[0].set_title('Training and validation accuracy and loss for best ' + model_name)
    axarr[0].set_ylabel('Accuracy')
    axarr[0].legend()
    #loss graph
    loss_r = results[ind1][5]
    val_loss_r = results[ind1][7]
    axarr[1].plot(epochs, loss_r, 'bo', label='Training loss')
    axarr[1].plot(epochs, val_loss_r, 'b', label='Validation loss')
    axarr[1].set_xlabel('Epochs')
    axarr[1].set_ylabel('Loss')
    axarr[1].legend()
    return

loss_acc_graphs_combined(6, 10, 'LSTM with RD')
loss_acc_graphs_combined(1, 10, 'GRU with RD')
loss_acc_graphs_combined(2, 10, 'Bidirectional with RD')



