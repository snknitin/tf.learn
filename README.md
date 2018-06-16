# tf.learn
My notes for learning to use TensorFlow for Deep Learning



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
    * Dot product - a.dot(b)
    * Matrix operations
        * np.linalg.inv(A)
        * np.linalg.norm(a)
        * np.linalg.det(A)
        * np.diag(A) - if you pass a 2d array you get a 1d array of diagonal elements but if you pass a 1d array, you get a diagonal matrix with that array elements
        * np.outer(a,b)
        * np.inner(a,b)
        * np.trace(A)  - sum of diagonal elements
        * cov = np.cov(X.T) - covariance. Calculate it so that we have (num_col,num_col) shape , so transpose
        * np.linalg.eigh(cov) and np.linalg.eig(cov)
     * Solving linear system of equations
        * Ax=B cna be solves by multiplying with A.inv
        * x= np.linalg.solve(A,B)
        * This is more efficient that calculating inverse and multiplying
    * Fourier transform ( converts signal from time domain to frequency domain ie., show the frequency components)
        
            x = np.linspace(0,100,1000)
            y= np.sin(x) + np.sin(3x) + np.sin(5x)
            Y= np.fft.fft(y)
            plt.plot(np.abs(Y))
            
* Pandas
    * Dataframe has a method as_matrix() which converts it into a matrix to be used in numpy
    * **df.sample(n=250)** to sample random rows from the dataframe
    * They have inbuilt plot functionalities as well. df.plot(x,y,kind='scatter') will create a scatter plot between two columns x and y
    * **df[cols\_to\_norm].apply(lambda x:(x - x.min()) / (x.max() - x.min()))** to normalize specific columns
    * **pd.get_dummies(y_train).as_matrix()** will create one hot encodings automatically as a numpy matrix.
    * For Joins use pd.merge(df1,df2,on='<insert_column_name>')
    
* Matplotlib
    * To see the visualizations we use **%matplotlib inline** in Jupyter
    * xlim and ylim functions set the limits for the range of axes in plots
    * Visualize a matrix using **plt.imshow(mat,cmap='')** and use a particular colormap 
    * **plt.colorbar()** will give you a legend for the imshow colormap 
    * Use label parameter if you have multiple plots in the same figure. That way you can call **plt.legend()**
    * **plt.tight_layout()** so that hte legend box doesn't block anything
    * To plot in 3d :
   
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection ='3D')
            ax.scatter(data_x,data_y,data_z,c=data[1])
    
* Scikit-learn
    * This package supports everything except neural networks
    * Load Datasets
        * from sklearn.datasets import <dataset_name>
        * load_data()
    * **MinMaxScaler** can be used to fit and transform the data for normalization to [0,1] range
        * from sklearn.preprocessing import MinMaxScaler
        * scaler = MinMaxScaler()
        * scaled_x_train = scaler.fit_transform(X_train)

    * Train test split can be done easily
        * from sklearn.model_selection import **train_test_split**
        * X_train, X_test, y_train, y_test = train_test_split(feat_data,
                                                    labels,
                                                    test_size=0.3,
                                                   random_state=101)
    
    * Classification evaluation metrics
        * from sklearn.metrics import confusion_matrix,classification_report
        * print(classification_report(predictions,y_test))
   
 * **Scipy**
 
    * Always use scipy for probability distributions sincce it is faster
 
             from scipy.stats import norm
             norm.pdf(0)
             norm.pdf(0,loc=5,scale=10) - loc is mean and scale is stddev  
             norm.pdf(np.random.rand(10)) - calculate pdf of all values in array  

             norm.logpdf() - log pdf  
             norm.cdf() - cumulative dist func  
             norm.logcdf() 
 
    * Sample from a Gaussian : Spherical(2d with mean 0 and variace 1, each dimension is uncorrelated and independent of each other due to unit variance)

            r= np.random.rand(10000,2)
            r = 5(r)+10

    * Sample from a general multivariate normal distribution where dimensions are not necessarily independent(full covariance matrix)

            cov = np.array([[1,0.8],[0.8,1]])
            from scipy.stats import multivariate_normal as mvn
            mu =  np.array([0,2])
            r = mvn.rvs(mean=mu,cov=cov,size=1000)
            plt.scatter(r[:,0],r[:,1])
            plt.axis('equal')
            
     * Loading matlab files or audio files
        * Loading .mat files    
        
                scipy.io.loadmat
                
         * Audio files contain signal amplitude at every point where sound is samples and the typical sampling rate is 44.1kHz(44100 samples(integers) for every second of sound)  
            
                scipy.io.wavfile.read
                scipy.io.wavfile.write
        * Convolution for signal processing
            
                 scipy.signal.convolve
                 scipy.signal.convolv2d   for images
                 
* SpaCy and Cython

    * It mixes C/C++ with Python, in a easy to learn interface.
    * The main reason is that Cython gives you a compiled code and the discussion is related to compilers and interpreters.In Cython the code needs to be “executable”. A compiler (e.g. gcc, cpp, icc) reads and analyses all the source code and produce a lower machine language of the code you have written as an executable file.

    * **Profiling** - You should thus start by profiling your Python code and find where the slow parts are located.
    
            import cProfile
            import pstats
            import my_slow_module
            cProfile.run('my_slow_module.run()', 'restats')
            p = pstats.Stats('restats')
            p.sort_stats('cumulative').print_stats(30)

    * 

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
    * You get a error if you try to run a variable without initializing. Use **init = tf.global_variables_initialzer()** and sess.run(init)
    * **initializer = tf.variance_scaling_initializer()** allows the intializer to adapt to the size of the weight tensors. This is useful for stacked autoencoders. Now initialize weights for each layer with the initializer
    * Variable scope - Allows modular sections or subsets of parameters to reuse that module
            
            with tf.varaible_scope('gen',reuse=True):
                
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
    * tf.train.Saver(var_list)


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

![alt text](https://github.com/snknitin/tf.learn/blob/master/static/LSTM2.png)
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

Use output projection wrapper to get the correct dimension of output:  
**cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units,activation = tf.nn.relu), output_size)**  
**output, states = tf.nn.dynamic_rnn(cell,X, dtype=tf.float32)**  

## Word2Vec using Gensim
    
the reason we do this is because the text data is usually sparse. Use gensim instead of tf. Words in a sentence can be thought of as a sequence. In NLP techniques words are usually replaced with numbers indicating some frequency relationship to their documents but in doing so we lose important information about the relationship between the words themselves.  

* Count base -  compute the statistics of how oftern some words co occur with neighbors in a large text corpus and mapt these to a small dense vector
* Predictive based  - directly try to predict the word from its nbrs in terms of learned small dense embedding vectors
    Pros : 
    * Similar words end up being close together
    * The model may produce axes that represent concepts such as gender, vers, singular vs plural etc
    * CBOW - The dog chews the **bone**
        * Takes source context words and tries to find best prediction of target word
        * Best for smaller datasets since bag of words smoothes over a lot of the distributional info by treating an entire context as one observation
        * It uses a binary classification objective like **logistic regression** where we compare the target word with rest of the noise words
        ![alt text](https://github.com/snknitin/tf.learn/blob/master/static/word2veccbow.PNG)
        * **Training is Noise-Contrastive**. We draw k words from noise distribution to make it computationally efficient
        * Assign high probability to correct words and low for noise words
        * Visualize these by reducing dimensions from 150 to 2 by  using **t-Distributed Stochastic Neighbor Embedding**
    * Skipgram - 
        * Predicts source context words from the target word
        * Best for large datasets
     
* Coding :
    * init_embeds = tf.random.uniform([vocabulary_size,Embedding_size],-1.0,1.0)
    * embeddings = tf.Variable(init_embeds)
    * embed = tf.nn.embedding_lookup(embeddings,train_inputs)
    * nce_weights - tf.Variable(tf.truncated_normal([vocab_size,embedding_size],stddev=1.0/np.sqrt(embedding_size)))
    * nce_biases = tf.Variable(tf.zeros([vocab_size]))
    * loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights,nce_biases,train_labels,embed,num_sampled,vocab_size))
    



# TensorFlow Abstractions


Some sort of higher level API that allows you to simplify the code yet still call the same sort of tensorflow commands. Eg: Skipping placeholders,feed dictionaries etc.,  Perfect when all you want to do is stack layers on top of each other


* **Keras API** : It became a part of tensorflow and can be called from it
    * from tensorflow.contrib.keras import models
    * **dnn_model = models.Sequential()**
    * from tensorflow.contrib.keras import **layers**
    * Adding Layers and Stacking
        * **dnn_model.add(layers.Dense(units=num_features,input_dim=,activation='relu'))** , repeat this line for stacking but remove the input_dim parameter for those
    * Last layer -  dnn_model.add(layers.Dense(units=num_output_classes,**activation='softmax'**))
    * To see what activations are available
        * from tensorflow.contrib.keras import **losses,optimizers,metrics,activations**
    * Final step is to compile
        * dnn_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        * If data is not one-hot coming in we use sparse else skip the sparse part
    * Now fit the model
        * dnn_model.fit(scaled_x_train,y_train, epochs=50)
    * Making predictions
        * dnn_model.predict_classes(scaled_x_test)
     
* **Layers API**

Layers module in tf.layers , which got past the tf.contrib.layers and became a feature of tensorflow.

* To get slightly more options for fully connected layers we use contrib
    * from tf.contrib.layers import fully_connected
    * hidden1 = fully_connected(X,num_hidden1,activation_fn=tf.nn.relu)
    * hidden2 = fully_connected(hidden1,num_hidden2,activation_fn=tf.nn.relu)
    * output = fully_connected(hidden2,num_outputs)
    * loss =  tf.losses.softmax_cross_entropy(onehot_labels=y_true,logits=output)
* To perform **PCA using Autoencoders**, the activation is fully_connected is set to None
    
# Tensorboard

Create a graph

    with tf.Session() as sess:  
        writer = tf.summary.FileWriter("./output",sess)
        print(sess.run())
        writer.close()

To view the graph   
**tensorboard --logdir="./output"**

* Every tensorflow operation has a parameter called name. Spaces are not allowed in it. we can also add scopes
* **with tf.name_scope("Operation_A"):** and every variable named under this is encompassed. We can also have nested scopes
* To merge all your summaries into one node
        
        summaries = tf.summary.merge_all()
        writer.add_summary(sess.run(...),global_step=step)

For more info [click here](https://www.tensorflow.org/versions/r1.1/get_started/summaries_and_tensorboard)








