import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        ## Create the torch.nn.Parameter for the weights and bias (if bias=True)
        ## Be careful about the size
        # ----- CHECK -----
        if(bias == True):
            self.W = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
            self.b = torch.nn.Parameter(torch.randn(out_channels))
        else:
            self.W = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
            self.b = None # We should not even initialize bias because we don't want it registered as a parameter for the model to tune.
    
    def __call__(self, x):
        
        return self.forward(x)


    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """

        # call MyFConv2D here
        # ----- TODO ----- # TODO TODO TODO -> NEED TO VERIFY!!!

        batch_size = x.shape[0]
        out_channels = self.out_channels # this should be the number of filters.
        input_height = x.shape[2]
        input_width = x.shape[3]

        # Before I used a bunch of for loops in order to perform the convolution.
        # This ended up taking my code forever to load, and it was taking me forever to run
        # my CNN for just a few epochs. After doing some research, I learned that 
        # it's much more efficent to use built-in pytorch functions to avoid having to
        # use four loops to perform a convolution (super naive approch).

        # Without any for loops, you can just precompute all the patches that would be present
        # if you did the for loop normally with F.unfold.
        all_x_windows = F.unfold(x, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)
        # all_x_windows now has dims (batch_size, 3 * kernel_size * kernel_size, output_height * output_width)
        # every column (dim = 2) now represents all the values in a single patch

        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # since I am going to multiply self.weights by the patches, I need to make sure that
        # every row is a patch, since this way, the dimensions will line up for matrix multiplication as follows:
        # patch * weights = output => (num_patches x window_size) x (window_size x num_filters) = (num_patches x num_filters)

        transposed_windows = all_x_windows.transpose(1, 2) # now, every row (dim = 1) all the values in a patch.

        # to get the weights in form (window_size x num_filters), I need to flatten them:
        
        flattened_weigths = self.W.flatten(start_dim=1) # get in size (num_filters x window_size)
        alligned_weigths = flattened_weigths.t() # tranpose to arrive at (window_size x num_filters)

        # now it is possible to perform the following operation:
        # patch * weights = output => (num_patches x window_size) x (window_size x num_filters) = (num_patches x num_filters)

        windows_x_weights = transposed_windows.matmul(alligned_weigths) # (num_patches x num_filters)

        # now, I just need to reshape the output to return the same as would result from a large for-loop:
        
        # goal: (batch_size, out_channels, output_height, output_width)
        output = windows_x_weights.transpose(1, 2) # now (num_filters (self.out_channels) x num_patches)
        # num_patches can now be unrolled into output height and width.
        output = output.view(batch_size, self.out_channels, output_height, output_width) 

        if self.b is not None:
          output += self.b.view(1, -1, 1, 1) # we want to reshape b in terms of its output channels

        return output
    
    # previous attemptt hat was too slow:

          # # call MyFConv2D here

        # batch_size = x.shape[0]
        # out_channels = self.out_channels # this should be the number of filters.
        # input_height = x.shape[2]
        # input_width = x.shape[3]

        # # ADD THE PADDING!!!:
        # X_with_padding = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode = "constant", value = 0)


        # output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        # output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # output = torch.zeros(batch_size, out_channels, output_height, output_width, device = x.device)

        # for b in range(batch_size):
        #     for h in range(output_height):
        #         for w in range(output_width):
        #             h_start = h * self.stride
        #             h_stop = h_start + self.kernel_size
        #             w_start = w * self.stride
        #             w_stop = w_start + self.kernel_size

        #             x_of_interest = X_with_padding[b] # We want x in form (in_chan, h, w) given the batch_cur

        #             x_of_interest = x_of_interest[:, h_start: h_stop, w_start: w_stop] # x is (3, 3, 3)

        #             # w is in (out_chan, in_chan, 3, 3)
        #             dotprod = torch.tensordot(x_of_interest, self.W, dims = ([0, 1, 2], [1, 2, 3]))
        #             # dotprod is in shape (1,) => [x1, x2,... x8]

        #             # the output channel number is taken care of from the convolution.
        #             if self.b is not None:
        #                 output[b, :, h, w] = dotprod + self.b # due to shape of weights, the output already has the proper dims.
        #             else:
        #                 output[b, :, h, w] = dotprod
    
    

    
class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size

        ## Take care of the stride
        ## Hint: what should be the default stride_size if it is not given? 
        ## Think about the relationship with kernel_size
        # ----- CHECK -----

        if(stride == None):
            self.stride = 1
        else:
            self.stride = stride

    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """

        x = x.contiguous() # save on memeory here.
        
        ## check the dimensions
        self.batch_size = x.shape[0]
        self.channel = x.shape[1] # in_channels = out_channels
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]
        
        ## Derive the output size
        # ----- CHECK -----

        output_height = (self.input_height - self.kernel_size) // self.stride + 1 # we want to use integer division here.
        output_width = (self.input_width- self.kernel_size) // self.stride + 1

        self.output_height   = output_height
        self.output_width    = output_width

        ## Maxpooling process
        ## Feel free to use for loop
        # ----- CHECK -----
        
        # exact same process with the conv2d. we want to precompute the windows rather than use a giant for-loop
        all_x_windows = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        # all_x_windows now has dims (batch_size, 3 * kernel_size * kernel_size, output_height * output_width)
        # every column (dim = 2) now represents all the values in a single patch 

        # For every window, we want the exact same dimensions, except now we want every row to correspond to 
        # all the elements within the window, so we can just take the max of that.
        num_elements_in_window = self.kernel_size * self.kernel_size

        all_x_windows = all_x_windows.view(self.batch_size, self.channel, num_elements_in_window, -1)

        # Now, we just take the max value of the second dimension, which grabs the highest value fromm the second dimension
        # we completely avoid using a for loop now! horray! 
        max_from_each_pool = all_x_windows.max(dim=2)[0]

        # now, we just need to return the max values from each pool in the proper output shape.
        output = max_from_each_pool.view(self.batch_size, self.channel, output_height, output_width)

        return output
    
    # previously attempt that was too slow:

    # def forward(self, x):

    #         """
    #         [input]
    #         x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

    #         [output]
    #         output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

    #         [hint]
    #         * out_channel == in_channel
    #         """

    #         print("mpool beg")

    #         ## check the dimensions
    #         self.batch_size = x.shape[0]
    #         self.channel = x.shape[1]
    #         self.input_height = x.shape[2]
    #         self.input_width = x.shape[3]

    #         ## Derive the output size
    #         # ----- CHECK -----

    #         output_height = (self.input_height - self.kernel_size) // self.stride + 1 # we want to use integer division here.
    #         output_width = (self.input_width- self.kernel_size) // self.stride + 1

    #         self.output_height   = output_height
    #         self.output_width    = output_width
    #         self.output_channels = self.channel
    #         self.x_pool_out      = self.batch_size

    #         ## Maxpooling process
    #         ## Feel free to use for loop
    #         # ----- CHECK -----

    #         output = torch.zeros(self.x_pool_out, self.output_channels, self.output_height, self.output_width, device = x.device)

    #         # currently, this does not account for the strides.

    #         for b in range(self.batch_size):
    #             for c in range(self.output_channels):
    #                 for w in range(output_width): # we do not want to skip by stride because our final output SHOULD be in the correct shape.
    #                     for h in range(output_height):
    #                         w_start = w * self.stride
    #                         w_end = w_start + self.kernel_size
    #                         h_start = h * self.stride
    #                         h_end = h_start + self.kernel_size

    #                         output[b, c, h, w] = torch.max(x[b, c, h_start: h_end, w_start: w_end])

    #         print("mpool end")

    #         return output
        

    
class FCNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out): # n_out should be 100. would do CE loss.
        super().__init__()

        self.input = nn.Linear(n_in, n_hidden)
        self.act1 = nn.ReLU()
        self.hidden = nn.Linear(n_hidden, n_hidden)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(n_hidden, n_out)

        # self.final_act = nn.Sigmoid()
        
    def forward(self, x): # first dimension is batch
        x = x.view(x.size(0), -1)
        # print(x)
        x = self.input(x)
        x = self.act1(x)
        x = self.hidden(x)
        x = self.act2(x)
        x = self.output(x)
        # x = self.final_act(x) # Cross Entropy Loss expects logits.
        return x
    

if __name__ == "__main__":


        
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle = True, num_workers = 2) # num_workers is used to speed up training via preloading
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle = False, num_workers = 2)

    sample_image, sample_label = train_dataset[0]
    print(sample_image.size())



    # %% [markdown]
    # # Test Cases

    # %%
    ########################################################################
    # MAX POOL 2D:
    print("MAX POOL TEST CASES:")


    MY_maxpool = MyMaxPool2D(kernel_size = 2, stride = 2)
    PYTORCH_maxpool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

    # TEST 1

    mp_test_1_image, _  = train_dataset[0]
    mp_test_1 = mp_test_1_image.unsqueeze(0).clone().detach() # we want to add the batch dimension so it's readable by both functions.

    MY_maxpool_pred = MY_maxpool(mp_test_1)
    PYTORCH_maxpool_pred = PYTORCH_maxpool(mp_test_1)


    if(torch.equal(MY_maxpool_pred, PYTORCH_maxpool_pred)):
        print("TEST 1: PASS. Micah's maxpool produces the same result as the official implementation.")
    else:
        print("SADNESS!")

    # TEST 2

    MY_maxpool = MyMaxPool2D(kernel_size = 4)
    PYTORCH_maxpool = torch.nn.MaxPool2d(kernel_size = 4)

    mp_test_2_image, _  = train_dataset[1] # new image for testing.
    mp_test_2 = mp_test_2_image.unsqueeze(0).clone().detach() # we want to add the batch dimension so it's readable by both functions.

    MY_maxpool_pred2 = MY_maxpool(mp_test_2)
    PYTORCH_maxpool_pred2 = PYTORCH_maxpool(mp_test_2)

    if(torch.equal(MY_maxpool_pred, PYTORCH_maxpool_pred)):
        print("TEST 2: PASS. Maybe his maxpool wasn't a fluke!")
    else:
        print("GREAT SADNESS!")

    ########################################################################
    # Convolutional Layer:
    print("CONVOLUTIONAL LAYER TEST CASES:")

    # at first, my tests were all failing until I realized that the randomly
    # initialized weights between nn.Conv2d were of course different form the 
    # weights initialized form my own function. To work around this, I just used
    # the F.conv2d function and created an output directly, using the exact same
    # weights for both functions. To ensure complete fairness, I initalized the weights
    # in the exact same way I would have within my own conv2d function, independent
    # of my function (I initialize my weights to ensure independence.)
    # I initialized with my conv2d function. This is how I make a fair comparison
    # between my implementation and the official implementation.

    # TEST 1

    test_1_params = [64, 3, 3, 2, 1]
    out_channels, in_channels, kernel_size, stride, padding = test_1_params
    FAIR_WEIGHTS = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
    FAIR_BIAS = torch.nn.Parameter(torch.randn(out_channels))

        # def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

    conv_test_1_image, _  = train_dataset[0]
    conv_test_1 = conv_test_1_image.unsqueeze(0).clone().detach()

    MY_conv_layer = MyConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    with torch.no_grad():
        MY_conv_layer.W.copy_(FAIR_WEIGHTS)
        if(MY_conv_layer.b is not None):
            MY_conv_layer.b.copy_(FAIR_BIAS)

    MY_conv2d_pred = MY_conv_layer(conv_test_1)
    PYTORCH_conv_pred = F.conv2d(conv_test_1, weight=FAIR_WEIGHTS, bias = FAIR_BIAS, stride=stride, padding= padding)


    if(torch.allclose(MY_conv2d_pred, PYTORCH_conv_pred, atol=1e-5)):  # add tolerance term due to differences in floating point math
        print("TEST 1: PASS. With identical starting weights, the functions produce the same convolution.")
    else:
        print("SADNESS!")

    # TEST 2

    test_2_params = [100, 3, 2, 1, 1]
    out_channels, in_channels, kernel_size, stride, padding = test_2_params
    FAIR_WEIGHTS2 = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
    FAIR_BIAS2 = torch.nn.Parameter(torch.randn(out_channels))

        # def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

    conv_test_2_image, _  = train_dataset[1] # next image
    conv_test_2 = conv_test_2_image.unsqueeze(0).clone().detach()

    MY_conv_layer2 = MyConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    with torch.no_grad():
        MY_conv_layer2.W.copy_(FAIR_WEIGHTS2)
        if(MY_conv_layer2.b is not None):
            MY_conv_layer2.b.copy_(FAIR_BIAS2)

    MY_conv2d_pred2 = MY_conv_layer2(conv_test_2)
    PYTORCH_conv_pred2 = F.conv2d(conv_test_2, weight=FAIR_WEIGHTS2, bias = FAIR_BIAS2, stride=stride, padding= padding)


    if(torch.allclose(MY_conv2d_pred2, PYTORCH_conv_pred2, atol=1e-5)):  # add tolerance term due to differences in floating point math
        print("TEST 2: PASS. Maybe the conv2d wasn't a fluke either!")
    else:
        print("SADNESS!")

    # %% [markdown]
    # # FCNN Model

    # %%
    FCNN_model = FCNN(n_in = sample_image.numel(), n_hidden=500, n_out = 100) # setting n_out to 100. We already KNOW the parameters of the problem.
    loss_fc = torch.nn.CrossEntropyLoss() # by default, this should aggregate all the loss for the entire batch.
    FCNN_optimizer = torch.optim.Adam(params=FCNN_model.parameters(), lr = 0.001)

    FCNN_epochs = 30 # doesn't need to be insanely high, since the model sees so many images every epoch.
    train_accuracy_FCNN = []
    test_accuracy_FCNN = []
    train_loss_FCNN = []
    test_loss_FCNN = []

    for i in range(FCNN_epochs):
        cur_train_loss = []
        cur_train_accuracy = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            FCNN_model.train()

            FCNN_model_pred = FCNN_model(images)
            FCNN_loss = loss_fc(FCNN_model_pred, labels) # predictions first for loss_fc
            cur_train_loss.append(FCNN_loss.item())

            FCNN_optimizer.zero_grad()
            FCNN_loss.backward()
            FCNN_optimizer.step()

            FCNN_accuracy = (torch.argmax(FCNN_model_pred, dim=1) == labels).float().mean().item()
            cur_train_accuracy.append(FCNN_accuracy)
            
        cur_test_loss = []
        cur_test_accuracy = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                FCNN_model.eval()

                FCNN_model_pred = FCNN_model(images)
                FCNN_loss = loss_fc(FCNN_model_pred, labels) 
                cur_test_loss.append(FCNN_loss.item())

                FCNN_accuracy = (torch.argmax(FCNN_model_pred, dim=1) == labels).float().mean().item()
                cur_test_accuracy.append(FCNN_accuracy)

            # aggregate metrics

        train_avg_acc = np.average(cur_train_accuracy)
        train_accuracy_FCNN.append(train_avg_acc)
        test_avg_acc = np.average(cur_test_accuracy)
        test_accuracy_FCNN.append(test_avg_acc)

        train_avg_loss = np.average(cur_train_loss)
        train_loss_FCNN.append(train_avg_loss)
        test_avg_loss = np.average(cur_test_loss)
        test_loss_FCNN.append(test_avg_loss)

        print(f"Epoch {i+1} train loss for FCNN: {train_avg_loss:.4f}")
        print(f"Epoch {i+1} train accuracy for FCNN: {train_avg_acc:.4f}")
        print(f"Epoch {i+1} test loss for FCNN: {test_avg_loss:.4f}")
        print(f"Epoch {i+1} test accuracy for FCNN: {test_avg_acc:.4f}")

    # %%
    # FCNN PLOTS, MAX ACCURACY + MODEL ARCHITECTURE, AND HYPERPARAMETERS

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  
    ax[0].plot(range(FCNN_epochs), train_loss_FCNN, color='tab:blue', label="Train Loss")
    ax[0].set_title("FCNN Train Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(range(FCNN_epochs), test_loss_FCNN, color='tab:red', label="Test Loss")
    ax[1].set_title("FCNN Test Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    plt.show()


    max_train_acc = max(train_accuracy_FCNN)
    max_test_acc = max(test_accuracy_FCNN)

    print("Below is the FCNN model architecture")
    print(FCNN_model)

    print("Relevant hyperparameters:")
    print(f"Epochs: {FCNN_epochs}")
    print(f"Optimizer: {type(FCNN_optimizer).__name__}")
    print(f"Loss Function: {loss_fc.__class__.__name__}")
    print(f"Learning Rate: {0.001}")
    print(f"Activation Function: ReLU! I just wanted to go for simplicity")
    print("I also used a hidden layer of 500 neurons to make sure that the FCNN was capable of learning the CIFAR100 data to some extent.")
        
    print(f"The max FCNN training accuracy was {max_train_acc}.")
    print(f"The max FCNN testing accuracy was {max_test_acc}.")

    # %% [markdown]
    # # CNN Model

    # %%
    class MyCNN(nn.Module):
        def __init__(self, in_channels, out_channels, first_filter): # n_out should be 100. would do CE loss.
            super().__init__()
            k = first_filter # normally, you should double as you go deeper.

            kernel_size = 3
            self.out_channels = out_channels

            self.conv1 = MyConv2D(in_channels, k, kernel_size, stride = 2, padding = 1, bias = True) # manually added filter size # using stride = 2 to reduce the feature space
            self.act1 = nn.ReLU()
            self.maxpool1 = MyMaxPool2D(kernel_size, stride = 1)
            self.conv2 = MyConv2D(k, k*2, kernel_size, stride = 2, padding = 1, bias = True) # manually added filter size 
            self.act2 = nn.ReLU()
            self.maxpool2 = MyMaxPool2D(kernel_size, stride = 1)
            # self.conv3 = MyConv2D(k*2, k*4, kernel_size, stride = 2, padding = 1, bias = True) # manually added filter size 
            
            # get length of num_output_features
            with torch.no_grad():
                x = torch.zeros(1, in_channels, 32, 32) # use a tensor in same shape as sample to get the length of the feature vector.
                x = self.conv1(x)
                x = self.act1(x)
                x = self.maxpool1(x)
                x = self.conv2(x)
                x = self.act2(x)
                x = self.maxpool2(x)
                # x = self.conv3(x)
                # x = self.maxpool(x)
                x = x.view(x.size(0), -1) # get the batch dimension. Flatten the rest to get fully connected layers to the output.
                num_output_features = x.size(1)

            self.linear = nn.Linear(num_output_features, self.out_channels)

        def forward(self, x): # first dimension is batch
            x = self.conv1(x)
            x = self.act1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.maxpool2(x)
            # x = self.conv3(x)
            # x = self.maxpool(x)
            x = x.view(x.size(0), -1) # get the batch dimension. Flatten the rest to get fully connected layers to the output.
            # num_output_features = x.size(1)
            x = self.linear(x)
            return x

    #  Conv2D:        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
    #  Max Pooling:   def __init__(self, kernel_size, stride=None):

    # %%
    CNN = MyCNN(in_channels = 3, out_channels = 100, first_filter = 3)
    CNN_optimizer = torch.optim.Adam(params = CNN.parameters(), lr = 0.001)
    loss_fc = torch.nn.CrossEntropyLoss() # by default, this should aggregate all the loss for the entire batch.

    CNN_epochs = 30
    train_accuracy_CNN = []
    test_accuracy_CNN = []
    train_loss_CNN = []
    test_loss_CNN = []

    for i in range(CNN_epochs):
        cur_train_loss = []
        cur_train_accuracy = []
        

        for batch_idx, (images, labels) in enumerate(train_loader):
            # print(batch_idx)
            CNN.train()

            CNN_model_pred = CNN(images)
            CNN_loss = loss_fc(CNN_model_pred, labels) # predictions first for loss_fc
            cur_train_loss.append(CNN_loss.item())

            CNN_optimizer.zero_grad()
            CNN_loss.backward()
            CNN_optimizer.step()

            CNN_accuracy = (torch.argmax(CNN_model_pred, dim=1) == labels).float().mean().item()
            cur_train_accuracy.append(CNN_accuracy)
        
        cur_test_loss = []
        cur_test_accuracy = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                CNN.eval()
                CNN_model_pred = CNN(images)
                CNN_loss = loss_fc(CNN_model_pred, labels) 
                cur_test_loss.append(CNN_loss.item())

                CNN_accuracy = (torch.argmax(CNN_model_pred, dim=1) == labels).float().mean().item()
                cur_test_accuracy.append(CNN_accuracy)

        train_avg_acc = np.average(cur_train_accuracy)
        train_accuracy_CNN.append(train_avg_acc)
        test_avg_acc = np.average(cur_test_accuracy)
        test_accuracy_CNN.append(test_avg_acc)

        train_avg_loss = np.average(cur_train_loss)
        train_loss_CNN.append(train_avg_loss)
        test_avg_loss = np.average(cur_test_loss)
        test_loss_CNN.append(test_avg_loss)

        print(f"Epoch {i+1} train loss for CNN: {train_avg_loss:.4f}")
        print(f"Epoch {i+1} train accuracy for CNN: {train_avg_acc:.4f}")
        print(f"Epoch {i+1} test loss for CNN: {test_avg_loss:.4f}")
        print(f"Epoch {i+1} test accuracy for CNN: {test_avg_acc:.4f}")

    # %%
    # CNN PLOTS, MAX ACCURACY + MODEL ARCHITECTURE, AND HYPERPARAMETERS

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  
    ax[0].plot(range(CNN_epochs), train_loss_CNN, color='tab:blue', label="Train Loss")
    ax[0].set_title("CNN Train Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(range(CNN_epochs), test_loss_CNN, color='tab:red', label="Test Loss")
    ax[1].set_title("CNN Test Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    plt.show()


    max_train_acc = max(train_accuracy_CNN)
    max_test_acc = max(test_accuracy_CNN)

    print("Below is the CNN model architecture")
    print(CNN)

    print("Relevant hyperparameters:")
    print(f"Epochs: {CNN_epochs}")
    print(f"Optimizer: {type(CNN_optimizer).__name__}")
    print(f"Loss Function: {loss_fc.__class__.__name__}")
    print(f"Learning Rate: {0.001}")
    print(f"Activation Function: ReLU! ReLU is the standard for CNNs.")
    print("I went for two convolutions because I want higher and lower levels of features to be learned.")

    print(f"The max CNN training accuracy was {max_train_acc}.")
    print(f"The max CNN testing accuracy was {max_test_acc}.")

    # %% [markdown]
    # # Visualization of Predictions

    # %%
    images, labels = next(iter(test_loader))
    first5_images = images[5:10]
    first5labels = labels[5:10]

    FCNN_model.eval()
    with torch.no_grad():
        FCNN_pred_idx = FCNN_model(first5_images).argmax(dim=1) 

    plt.figure(figsize=(4, 11))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.imshow(first5_images[i].permute(1, 2, 0))  # We need the images in form (H, W, C) - not (C, H, W)
        plt.title(f"Actual Class:{test_dataset.classes[first5labels[i]]}; Predicted Class: {test_dataset.classes[FCNN_pred_idx[i]]}")
        plt.axis("off")

    plt.suptitle("FCNN Predictions")
    plt.show()


    CNN.eval()
    with torch.no_grad():
        CNN_pred_idx = CNN(first5_images).argmax(dim=1) 

    plt.figure(figsize=(4, 11))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.imshow(first5_images[i].permute(1, 2, 0))  # We need the images in form (H, W, C) - not (C, H, W)
        plt.title(f"Actual Class:{test_dataset.classes[first5labels[i]]}; Predicted Class: {test_dataset.classes[CNN_pred_idx[i]]}")
        plt.axis("off")

    plt.suptitle("CNN Predictions")
    plt.show()