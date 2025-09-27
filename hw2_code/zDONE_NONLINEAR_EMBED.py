from MLP_draft import *
from numpyNN import *

# NOTE: SEE END OF CODE TO TOGGLE BETWEEN RUNNING EACH MODEL

def adjusted_plot_decision_boundary(data_type, X, y, pred_fn, boundry_level=None, logs = None):
    """
    Plots the decision boundary for the model prediction
    :param X: input data
    :param y: true labels
    :param pred_fn: prediction function,  which use the current model to predict。. i.e. y_pred = pred_fn(X)
    :boundry_level: Determines the number and positions of the contour lines / regions.
    :return:
    """

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

    if(data_type == 'CIRCLE'):
        x2y2 = xx.ravel()**2 + yy.ravel()**2
        Z = pred_fn.predict(np.c_[xx.ravel(), yy.ravel(), x2y2])
    elif(data_type == 'XOR'):
        xy =  xx.ravel() * yy.ravel()
        Z = pred_fn.predict(np.c_[xx.ravel(), yy.ravel(), xy])
    else:
        sqx2y2 = np.sqrt(xx.ravel()**2 + yy.ravel()**2)
        arctan = np.arctan2(yy.ravel(), xx.ravel())
        Z = pred_fn.predict(np.c_[xx.ravel(), yy.ravel(), sqx2y2, arctan])

    print("printed", Z)
    Z = Z.reshape(xx.shape)

    plt.subplot(1, 2, 2)
    plt.xlabel(f"epochs: {logs['epochs'][-1]} | train loss: {logs['train_loss'][-1]:.3f} | val loss: {logs['val_loss'][-1]:.3f} | val acc: {logs['accuracy']:.1f}%", fontsize=10)
    plt.title('validation decision boundary',fontsize=15)
    plt.contourf(xx, yy, Z, alpha=0.7, levels=boundry_level, cmap='viridis_r')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), alpha=0.7,s=50, cmap='viridis_r',)

# CIRCLE DATA EXAMPLE
def non_linear_circle():
    X, y = circleData() # shape of X is (400, 2). We only are feeding one coord at a time. y is in (400, 1). len is first dim.

    # print(np.shape(X[:,1]))
    x2_y2 = X[:,0] ** 2 + X[:,1] ** 2
    # print(np.shape(x2_y2))

    sample_num = 400
    indices = np.random.permutation(sample_num)  # Shuffle the indices
    X_train, y_train = X[indices[:sample_num//2]], y[indices[:sample_num//2]]
    X_val, y_val = X[indices[sample_num//2:]], y[indices[sample_num//2:]]

    x2_y2_train = (X_train[:,0] ** 2 + X_train[:,1] ** 2).reshape(-1,1)
    x2_y2_val = (X_val[:,0] ** 2 + X_val[:,1] ** 2).reshape(-1,1)

    X_train = np.append(X_train, x2_y2_train, axis = 1)
    X_val = np.append(X_val, x2_y2_val, axis = 1)

    print("X_train", np.shape(X_train))
    print(type(X_train))
    print("y_train", np.shape(y_train))
    print("X_val", np.shape(X_val))
    print("y_val", np.shape(y_val))

    input_dim = np.shape(X_train)[-1]; output_dim = np.shape(y_train)[-1]
    hidden_dims = [500, 100]
    activations = ['Linear', 'Linear', 'Sigmoid'] # Keep last layer linear for regression and classificaiton
    init = 'Xavier' # 'He' or 'Xavier' is alternative
    loss_type = 'L2' # primary culprit loss, also num of epochs.
    # last layer of model should be sigmoid

    model = MyMLP(init = init, input_dim = input_dim, hidden_dims=hidden_dims, output_dim=output_dim, activations = activations)
    w_means = [np.mean(w_sub) for w_sub in model.w]
    print("FIRST w_means", w_means)

    train_loss = []
    val_loss = []
    epochs_list = []


    def train(model, X_train, y_train, X_val, y_val, epochs = 40000, lr = 0.0001):
        final_accuracy = 0
        for epoch in range(epochs):
            pred_y = model.forward(X_train)
            train_loss_i, train_grad = model.loss_fc(y = y_train, y_pred = pred_y, loss_type = loss_type)
            dLdw, dLdb = model.backward(train_grad)
            model.SGD_update(lr = lr)

            for_pred_y_val = model.forward(X_val)
            val_loss_i, val_grad = model.loss_fc(y = y_val, y_pred = for_pred_y_val, loss_type = loss_type)


            train_loss.append(train_loss_i)
            val_loss.append(val_loss_i)
            epochs_list.append(epoch + 1)

        pred_val = model.predict(X_val)
        final_accuracy = 100 * np.sum(pred_val == y_val)/len(y_val)
        return final_accuracy

    final_accuracy = train(model, X_train, y_train, X_val, y_val)

    print("shape", np.shape(final_accuracy))

    logs = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epochs': epochs_list,
                'accuracy': final_accuracy
        }

    print("X_Val", np.shape(X_val[:, :2]))
    print("y_Val", np.shape(y_val))

    plot_loss(logs)
    adjusted_plot_decision_boundary('CIRCLE', X_val, y_val, pred_fn = model, boundry_level=None, logs = logs)
    plt.show()
    
# XOR EXAMPLE
def non_linear_XOR():
    X, y = XORData() # shape of X is (400, 2). We only are feeding one coord at a time. y is in (400, 1). len is first dim.
    sample_num = 400
    indices = np.random.permutation(sample_num)  # Shuffle the indices
    X_train, y_train = X[indices[:sample_num//2]], y[indices[:sample_num//2]]
    X_val, y_val = X[indices[sample_num//2:]], y[indices[sample_num//2:]]

    xy_train = (X_train[:,0] * X_train[:,1]).reshape(-1,1)
    xy_val = (X_val[:,0] * X_val[:,1]).reshape(-1,1)

    X_train = np.append(X_train, xy_train, axis = 1)
    X_val = np.append(X_val, xy_val, axis = 1)

    print("X_train", np.shape(X_train))
    print(type(X_train))
    print("y_train", np.shape(y_train))
    print("X_val", np.shape(X_val))
    print("y_val", np.shape(y_val))

    input_dim = np.shape(X_train)[-1]; output_dim = np.shape(y_train)[-1]
    hidden_dims = [4, 2] # [2, 1] works, too, but it's less consitent.
    activations = ['Tanh', 'Linear', 'Sigmoid'] # Keep last layer linear for regression and classificaiton
    init = 'He' # 'He' or 'Xavier' is alternative
    loss_type = 'L2' # primary culprit loss, also num of epochs.
    # last layer of model should be sigmoid

    model = MyMLP(init = init, input_dim = input_dim, hidden_dims=hidden_dims, output_dim=output_dim, activations = activations)
    w_means = [np.mean(w_sub) for w_sub in model.w]
    print("FIRST w_means", w_means)

    train_loss = []
    val_loss = []
    epochs_list = []


    def train(model, X_train, y_train, X_val, y_val, epochs = 100000, lr = 0.0001):
        final_accuracy = 0
        for epoch in range(epochs):
            pred_y = model.forward(X_train)
            train_loss_i, train_grad = model.loss_fc(y = y_train, y_pred = pred_y, loss_type = loss_type)
            dLdw, dLdb = model.backward(train_grad)
            model.SGD_update(lr = lr)

            for_pred_y_val = model.forward(X_val)
            val_loss_i, val_grad = model.loss_fc(y = y_val, y_pred = for_pred_y_val, loss_type = loss_type)


            train_loss.append(train_loss_i)
            val_loss.append(val_loss_i)
            epochs_list.append(epoch + 1)

        pred_val = model.predict(X_val)
        final_accuracy = 100 * np.sum(pred_val == y_val)/len(y_val)
        return final_accuracy

    final_accuracy = train(model, X_train, y_train, X_val, y_val)

    print("shape", np.shape(final_accuracy))

    logs = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epochs': epochs_list,
                'accuracy': final_accuracy
        }

    print("X_Val", np.shape(X_val[:, :2]))
    print("y_Val", np.shape(y_val))

    plot_loss(logs)
    adjusted_plot_decision_boundary('XOR', X_val, y_val, pred_fn = model, boundry_level=None, logs = logs)
    plt.show()

# SWISS ROLL EXAMPLE
def non_linear_SWISS_ROLL():
    X, y = swissrollData() # shape of X is (400, 2). We only are feeding one coord at a time. y is in (400, 1). len is first dim.
    sample_num = 400
    indices = np.random.permutation(sample_num)  # Shuffle the indices
    X_train, y_train = X[indices[:sample_num//2]], y[indices[:sample_num//2]]
    X_val, y_val = X[indices[sample_num//2:]], y[indices[sample_num//2:]]

    sqx2_y2_train = (np.sqrt(X_train[:,0] **2 + X_train[:,1] ** 2)).reshape(-1,1)
    sqx2_y2_val =  (np.sqrt(X_val[:,0] **2 + X_val[:,1] ** 2)).reshape(-1,1)
    arctan_train = (np.arctan2(X_train[:,1], X_train[:,0])).reshape(-1,1)
    arctan_val =  (np.arctan2(X_val[:,1], X_val[:,0])).reshape(-1,1)


    X_train = np.append(X_train, sqx2_y2_train, axis = 1)
    X_train = np.append(X_train, arctan_train, axis = 1)    
    X_val = np.append(X_val, sqx2_y2_val, axis = 1)
    X_val = np.append(X_val, arctan_val, axis = 1)  

    print("X_train", np.shape(X_train))
    print(type(X_train))
    print("y_train", np.shape(y_train))
    print("X_val", np.shape(X_val))
    print("y_val", np.shape(y_val))

    input_dim = np.shape(X_train)[-1]; output_dim = np.shape(y_train)[-1]
    hidden_dims = list(map(int, 0.2 * np.array([125, 75])))
    activations = ['ReLU', 'Tanh', 'Sigmoid']  
    # activations = ['Tanh', 'Linear', 'Sigmoid']  

    init = 'He' # 'He' or 'Xavier' is alternative
    loss_type = 'BCE' # primary culprit loss, also num of epochs.
    # last layer of model should be sigmoid

    model = MyMLP(init = init, input_dim = input_dim, hidden_dims=hidden_dims, output_dim=output_dim, activations = activations)
    w_means = [np.mean(w_sub) for w_sub in model.w]
    print("FIRST w_means", w_means)

    train_loss = []
    val_loss = []
    epochs_list = []


    def train(model, X_train, y_train, X_val, y_val, epochs = 10000, lr = 0.0001):
        final_accuracy = 0
        for epoch in range(epochs):
            pred_y = model.forward(X_train)
            train_loss_i, train_grad = model.loss_fc(y = y_train, y_pred = pred_y, loss_type = loss_type)
            dLdw, dLdb = model.backward(train_grad)
            model.SGD_update(lr = lr)

            for_pred_y_val = model.forward(X_val)
            val_loss_i, val_grad = model.loss_fc(y = y_val, y_pred = for_pred_y_val, loss_type = loss_type)


            train_loss.append(train_loss_i)
            val_loss.append(val_loss_i)
            epochs_list.append(epoch + 1)

        pred_val = model.predict(X_val)
        final_accuracy = 100 * np.sum(pred_val == y_val)/len(y_val)
        return final_accuracy

    final_accuracy = train(model, X_train, y_train, X_val, y_val)

    print("shape", np.shape(final_accuracy))

    logs = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epochs': epochs_list,
                'accuracy': final_accuracy
        }

    print("X_Val", np.shape(X_val[:, :2]))
    print("y_Val", np.shape(y_val))

    plot_loss(logs)
    adjusted_plot_decision_boundary('SWISSROLL', X_val, y_val, pred_fn = model, boundry_level=None, logs = logs)
    plt.show()

# non_linear_circle()
non_linear_XOR()
# non_linear_SWISS_ROLL()