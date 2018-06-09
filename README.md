# tf.learn
Learning to use TensorFlow for Deep Learning



## Installation of the Environment

Get the yml file from [here](https://www.dropbox.com/s/k4i3gmo0bvss7g7/linux_tfdl_env.yml?dl=0)

    conda env create -f linux_tfdl_env.yml
    source activate tfdeeplearning
    source deactivate tfdeeplearning
    
    
## Crash Course

* Numpy
    * **Arange** is like range , start and stop with a step size exclusing the stop
    * **linspace** creates a linearly spaced array with start, stop and number of elements in the array
    * random seed needs to be executed each time , otherwise it might lead to different random values
    * argmax and argmin will return the index of the array where the max and min elements reside
    * When using matrices, you can use a mask or filter value as mat > 5 which turns everything into Boolean
    * To get the actual values mat[mat>50] will give the values from this filter
* Pandas
    * Dataframe has a method as_matrix() which converts it into a matrix to be used in numpy
    * **df.sample(n=250)** to sample random rows from the dataframe
    * They have inbuilt plot functionalities as well. df.plot(x,y,kind='scatter') will create a scatter plot between two columns x and y
    * **df[cols\_to\_norm].apply(lambda x:(x - x.min()) / (x.max() - x.min()))** to normalize specific columns
    
* Matplotlib
    * To see the visualizations we use **%matplotlib inline** in Jupyter
    * xlim and ylim functions set the limits for the range of axes in plots
    * Visualize a matrix using **plt.imshow(mat,cmap='')** and use a particular colormap 
    * **plt.colorbar()** will give you a legend for the imshow colormap 
    * Use label parameter if you have multiple plots in the same figure. That way you can call plt.legend()
    * plt.tight_layout() so that hte legend box doesn't block anything
* Scikit-learn
    * **MinMaxScalar** can be used to fit and transform the data for normalization to [0,1] range
    * Train test split can be done easily
    * This package supoorts everything except neural networks


## OOP Concept

When you extend a class or inherit from a super class, to access the first class's init method we use super().\_\_init\_\_(args) to access the super class 

## TensorFlow

* Syntax or useful commands
    * **tf.fill((dimensions),scalar value)** or use tf.ones and tf.zeros without the scalar parameter
    * **tf.random_normal((dimensions),mean,stddev)**
    * **tf.random_uniform((dimensions),minval,maxval)**
    * Call **sess =tf.InteractiveSession()** to evaluate operations outside the session. Useful in Jupyter notebook
    * a.get_shape() will give the TensroShape with dimensions
    * **tf.matmul(a,b** Matrix multiplication
    * **tf.multiply(a,b)** is element-wise multiplication
    * [ONLY FOR GPU USAGE](https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory)
        * gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        * with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
* Graphs
    * when you start TF, a default graph is created which can be accessed as a GraphObject using tf.get_default_graph()
    * g=tf.Graph() to create another graph
    * **with g.as_default():** This makes it the default
    * tf.reset_default_graph() to refresh 
 * Variables and Placeholders
    * These are the two types of Tensor objects
    * Placeholders are initially empty but they need to be declared with an expected data type(like tf.float32) and shape of the data
    * You get a error if you try to run a variable without initializing. Use init = tf.global_variables_initialzer() and sess.run(init)
 * Regression and Classificaton tasks
    * **error = tf.reduce_sum()** and tf.reduce_mean() to get cost functions
    * **optimizer = tf.train.GradientDescentOptimizer(learning_rate)** and **train = optimizer.minimize(error)**
 
     * Estimator Object
        * tf.estimator is an API with several models which resembles scikit-learn ML models
        * Feature columns -  
            * **tf.feature_column.numeric_column('x',shape=[1])** Use column names maybe from pandas df
            * **assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])**
            * tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size=10)  when you don't know how many groups
            * tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80]) To create ranges or buckets 
        * Use train_test_split to get x_train, x_eval, y_train, y_eval
        * Input function -  For train and eval 
            * tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size,num_epochs,shuffle=True)
            * tf.estimator.inputs.pandas_input_fn(x_train,y_train,batch_size,num_epochs,shuffle=True)
        * Model :
            * model = tf.estimator.LinearClassifier(feature_columns,n_classes)
            * model.train(input_fn,steps=1000) This gives global step-wise loss
            * trained_metrics = model.evaluate(input_fn,steps)  This gives a loss, average_loss and global step
            * model.predict(input_fn)
            
         * If you use a DNNClassifier, categorical columns should become an embedding_column or indicator_column  
            * model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns,n_classes)
            * embedded_group_column = tf.feature_column.embedding_column(assigned_group, dimension=4)
* Save and Restore
    * saver = tf.train.Saver()
    * Towards the end use  saver.save(sess,'models/my_first_model.ckpt')
    * saver.restore(sess,'models/my_first_model.ckpt')


## Convolutional Neural Networks

* Xavier initialization - Draw weights from a Gaus sian or Uniform distribution with zero mean and specific variance equal to inverse of number of neurons feeding into that particuar neuron
* Flattening an image removes some of the 2d information such as relationship with neighbouring pixels
* cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
* Unlike a Densely connected NN where each unit is connected to every unit in the next layer,  CNN has each unit connected to a smller number of nearby units in the next layer. [Watch this](https://www.youtube.com/watch?v=JiN9p5vWHDY)
* Nearby pixels are much more correlated to each other so each CNN layer looks at an increasingly larger part of the image and having units only connected to nearby units aids invariance. So we have regularization and also a limited search size for weights of the convolution


![alt text](https://github.com/snknitin/tf.learn/blob/master/static/CNN.png)

* Initialize the weights as **tf.Variable(tf.truncated_normal(shape,stddev=0.1))**
* Initialize bias as **tf.Variable(tf.constant(0.1,shape=shape))**

* Convolution
    * x = [batch,H,W,Channels]
    * W = [filter_H, filter_W, Channels IN, Channels OUT]
    * **tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME'/'VALID')**  Strides in each dimension
* Pooling 
    * **tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')**
    * ksize is the size of the window for each dimension of the input tensor
    * strides is the stride for the sliding window for each dimension of input tensor
    * We do it on the H and W dimension
* ReLU - tf.nn.relu(conv2d(input,W)+b)
* Fully Connected Layer - tf.matmul(input_layer,W)+b
* tf.reshape(x,[-1,28,28,1])
* tf.nn.dropout(final_layer,keep_prob=0.6)
* Accuracy
    * tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
    * acc = tf.reduce_mean(tf.cast(matches,tf.float32))
* CiFAR - To plot the image we need to reshape the image samples to (32,32,3)
    * Data Shape = (10000,3,32,32)
    * X = **X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")**


## RNN and LSTM, Time Series

Useful for sequential data like sentences, audio, car trajectories, time series data etc. Normal neuron aggregates weighted inputs into an acctivation function to get an output. Recurrent neuron sends the output back to the neuron. This can be unrolled. Cells that are a function of the input from previous time step is called a memory cell. Each Recurrent neuron has two sets of weights(Wx for input and Wy for output of that original input X)

* Seq2Seq - Text generation
* Seq2Vec - Sentiment scores
* Vec2Seq -  captioning images


To avoid vanishing gradients we can do batch normalization, gradient clipping, or just change the activation function to leaky relu.
Alternatively we can shorten the timesteps used for prediction but that gets worse at predicting longer trends. RNN's have an inherent memory loss anyway due to information being lost in each timestep. Use GRU or LSTM

![alt text](https://github.com/snknitin/tf.learn/blob/master/static/LSTM.PNG)

* Forget Gate - 
    * Sigmoid , 1 means keep it and 0 means forget
    * Maybe when you want to speak about the new subject you want to forget the old subject attributes like gender pronouns
    * Input xt and Previous output ht-1
* Input Gate - 
    * What new information will you store in the cell state
    * First part is a sigmoid(input) and second part is tanh(candidate cell state C~t)
* Current Cell state -
    * Ct = Forget previous cell state + input state with candidate cell state
* Output State
    * Use sigmoid for previous output ht-1 and current input xt to get output gate ot
    * Tanh of output with cell state Ct is the final output ht


Peephole LSTM -  here we pass Ct-1 to each of the three gates   
Gated Recurrent Unit - Combines forget and input gate into an Update Gate, merges cell state and hidden state too. This resulting model is simpler than LSTM
![alt text](https://github.com/snknitin/tf.learn/blob/master/static/GRU.png)

**cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units,activation = tf.nn.relu), output_size)**  
**output, states = tf.nn.dynamic_rnn(cell,X, dtype=tf.float32)**  









