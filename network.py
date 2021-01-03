import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


NN_ARCHITECTURE = [
    {"input_dim": 4, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "relu"},
]


def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    # number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['w' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values


def relu(z):
    return np.maximum(0, z)

def relu_backward(da, z):
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0
    return dz


def single_layer_forward_propagation(a_prev, w_curr, b_curr, activation="relu"):
    z_curr = np.dot(w_curr, a_prev) + b_curr

    if activation == "relu":
        activation_function = relu
    else:
        raise Exception('Non-supported activation function')

    return activation_function(z_curr), z_curr


def full_forward_propagation(x, params_values, nn_architecture):
    memory = {}
    a_curr = x

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        a_prev = a_curr

        activ_function_curr = layer["activation"]
        w_curr = params_values["w" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        a_curr, z_curr = single_layer_forward_propagation(a_prev, w_curr, b_curr, activ_function_curr)

        memory["a" + str(idx)] = a_prev
        memory["z" + str(layer_idx)] = z_curr

    return a_curr, memory


def get_cost_value(y_hat, y):
    m = y_hat.shape[1]
    cost = -1 / m * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))
    return np.squeeze(cost)


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(y_hat, y):
    y_hat_ = convert_prob_into_class(y_hat)
    return (y_hat_ == y).all(axis=0).mean()


def single_layer_backward_propagation(da_curr, w_curr, b_curr, z_curr, a_prev, activation="relu"):
    m = a_prev.shape[1]

    if activation == "relu":
        backward_activation_func = relu_backward
    else:
        raise Exception('Non-supported activation function')

    dz_curr = backward_activation_func(da_curr, z_curr)
    dw_curr = np.dot(dz_curr, a_prev.T) / m
    db_curr = np.sum(dz_curr, axis=1, keepdims=True) / m
    da_prev = np.dot(w_curr.T, dz_curr)

    return da_prev, dw_curr, db_curr


def full_back_propagation(y_hat, y, memory, params_values, nn_architecture):
    grads_values = {}

    m = y.shape[1]
    y = y.reshape(y_hat.shape)

    da_prev = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]

        da_curr = da_prev

        a_prev = memory["a" + str(layer_idx_prev)]
        z_curr = memory["z" + str(layer_idx_curr)]

        w_curr = params_values["w" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        da_prev, dw_curr, db_curr = single_layer_backward_propagation(da_curr, w_curr, b_curr, z_curr, a_prev, activ_function_curr)

        grads_values["dw" + str(layer_idx_curr)] = dw_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["w" + str(layer_idx)] -= learning_rate * grads_values["dw" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values


def train(x, y, nn_architecture, epochs, learning_rate, verbose=True, callback=None):
    params_values = init_layers(nn_architecture, 2)

    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        y_hat, cashe = full_forward_propagation(x, params_values, nn_architecture)

        cost = get_accuracy_value(y_hat, y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(y_hat, y)
        accuracy_history.append(accuracy)

        grads_values = full_back_propagation(y_hat, y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        if i % 50 == 0:
            if verbose:
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if callback is not None:
                callback(i, params_values)

    return params_values


iris = datasets.load_iris()
x = iris['data']
y = iris['target']

TEST_SIZE = 0.1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=21)


params_values = train(np.transpose(x_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), NN_ARCHITECTURE, 250, 0.01)

y_test_hat, _ = full_forward_propagation(np.transpose(x_test), params_values, NN_ARCHITECTURE)

acc_test = get_accuracy_value(y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f} ".format(acc_test))
