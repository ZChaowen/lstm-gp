import lstm
import time
import matplotlib.pyplot as plt
import numpy as np


def plot_results(predicted_data, true_data):
    group_labels = ['2017-01', '2017-01', '2017-01', '2017-01', '2017-01']
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # plt.xaxis.set_major_formatter(ax.FuncFormatter(format_date))
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("range")
    plt.xticks([0, 100, 200, 300, 400],
               ['2017-01', '2017-03', '2017-05', '2017-07', '2017-09'])

    plt.grid(True)
    # plt.axis(['2017-01','2017-12', -0.10, 0.15])

    # plt.xticks(predicted_data,group_labels,rotation=0)
    plt.show()

    """
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend(['True', 'Predict'], loc='upper left')
    plt.savefig('E:\ML\lstm_end\data_result\predict.jpeg', bbox_inches='tight')
    plt.show()
    """


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


# 保存结果
def save_result(y_test, predicted_values):
    np.savetxt('lstm_gp\save\True_Stock.csv', y_test)
    np.savetxt('lstm_gp\save\predicted_Stock.csv', predicted_values)

# Main Run Thread
if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 1
    seq_len = 50
    print('> Loading data... ')
    X_train, y_train, X_test, y_test = lstm.load_data('C:\Prectice_project\lstm_end\stock_data\sp500.csv', seq_len,
                                                      True)  # sinwave.csv，sp500美股标普500指数
    print('> Data Loaded. Compiling...')

    model = lstm.build_model([1, 50, 100, 1])  # model = lstm.build_model([1, 50, 100, 1])

    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)

    predicted= lstm.predict_point_by_point(model, X_test)
    #predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    print('Training duration (s) : ', time.time() - global_start_time)
    #plot_results_multiple(predictions, y_test, 50)
    plot_results(predicted, y_test)
    #save_result(y_test=y_test, predicted_values=predicted)
