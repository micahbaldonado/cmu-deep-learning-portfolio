from MLP_draft import *
from numpyNN import *

# NOTE: SEE END OF CODE TO TOGGLE BETWEEN RUNNING EACH MODEL


X, y = swissrollData() # shape of X is (400, 2). We only are feeding one coord at a time. y is in (400, 1). len is first dim.

sample_num = 400
indices = np.random.permutation(sample_num)  # Shuffle the indices
X_train, y_train = X[indices[:sample_num//2]], y[indices[:sample_num//2]]
X_val, y_val = X[indices[sample_num//2:]], y[indices[sample_num//2:]]

print("X_train", np.shape(X_train))
print("y_train", np.shape(y_train))
print("X_val", np.shape(X_val))
print("y_val", np.shape(y_val))

#FIXME FIXME UP UP UP

input_dim = np.shape(X)[-1]; output_dim = np.shape(y)[-1]
hidden_dims = list(map(int, 2.2 * np.array([125, 100, 100, 75])))
activations = ['Tanh', 'Tanh', 'Tanh', 'Tanh', 'Sigmoid'] 
init = 'He' # 'He' or 'Xavier' is alternative
loss_type = 'BCE' # primary culprit loss, also num of epochs.
# last layer of model should be sigmoid

model = MyMLP(init = init, input_dim = input_dim, hidden_dims=hidden_dims, output_dim=output_dim, activations = activations)
w_means = [np.mean(w_sub) for w_sub in model.w]
print("FIRST w_means", w_means)

# train_loss','val_loss' and 'epochs'

train_loss = []
val_loss = []
epochs_list = []


def train(model, X_train, y_train, X_val, y_val, epochs = 4000, lr = 0.005):
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

plot_loss(logs)
plot_decision_boundary(X_val, y_val, pred_fn = model, boundry_level=None, logs = logs)
plt.show()


