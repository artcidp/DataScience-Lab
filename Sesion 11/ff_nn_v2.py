import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


g_train2=tf.Graph()
with g_train2.as_default():

    with tf.name_scope("inputs"):
        X = tf.compat.v1.placeholder(tf.float32, [None, 784],name='x_data') 
        Ydata = tf.compat.v1.placeholder(tf.float32, [None, 10],name='y_data') 
    
    with tf.name_scope("fitting_variables"):
        with tf.name_scope("weights"):
            W = tf.Variable(tf.ones([10,784]),name='w')
            tf.compat.v1.summary.histogram("wieghts",W)
        with tf.name_scope("bias"):
            b = tf.Variable(tf.ones([10]),name='b')
            tf.compat.v1.summary.histogram("biases",b)
    
    prod=tf.matmul(W,X,transpose_b=True) #transponemos X que cada dato sea una columna
    sumation=tf.transpose(a=tf.transpose(a=prod) + b) #sumar b verticalmente con broadcasting
    
    with tf.name_scope("outputs"):
        with tf.name_scope("model_output"):
            Ymod = tf.nn.softmax(tf.transpose(a=sumation),name="y_mod") 
        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=Ydata * tf.math.log(Ymod), axis=[1])) 
            tf.compat.v1.summary.scalar("cross_entropy",cross_entropy)
            
        train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(input=Ymod,axis=1), tf.argmax(input=Ydata,axis=1)) 
        
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32)) 
            tf.compat.v1.summary.scalar("accuracy",accuracy)
            
            
batch_size=100
no_of_epochs=20

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

with tf.compat.v1.Session(graph=g_train2) as sess2:
    
    summaries_dir="test_logs/"
    #writer = tf.summary.FileWriter(summaries_dir,sess2.graph)
    train_writer = tf.summary.FileWriter(summaries_dir + '/train',sess2.graph)
    valid_writer = tf.summary.FileWriter(summaries_dir + '/valid')
    test_writer = tf.summary.FileWriter(summaries_dir + '/test')

    merged=tf.compat.v1.summary.merge_all()
    #test_writer = tf.summary.FileWriter(summaries_dir)
    
    tf.compat.v1.global_variables_initializer().run()
    #calculo inicial de accuracy, perdida,etc.
    summary,loss=sess2.run([merged,cross_entropy],feed_dict={X: mnist.train.images,Ydata: mnist.train.labels})
    train_writer.add_summary(summary, 0)
    #print("Epoch:=",0,"; \t Epoch Loss:=",loss)
    
    summary,t_acc=sess2.run([merged,accuracy], feed_dict={X: mnist.train.images, Ydata: mnist.train.labels})#calculates accuracy across all data
    train_writer.add_summary(summary, 0)
    #print("Training Accuracy is", t_acc*100,"%")
    summary,v_acc=sess2.run([merged,accuracy], feed_dict={X: mnist.validation.images,Ydata:mnist.validation.labels})
    valid_writer.add_summary(summary,0)
    #print("Validation Accuracy is", v_acc*100,"%")
    summary,f_acc=sess2.run([merged,accuracy], feed_dict={X: mnist.test.images,Ydata:mnist.test.labels})
    test_writer.add_summary(summary,0)
    #print("Test Accuracy is", f_acc*100,"%")
    
    sum_data=np.array([0,loss,t_acc,v_acc,f_acc]) #agregar los datos de la epoca 0 en una lista
    
    for epoch in range(no_of_epochs):
        epoch_loss=0
        for i in range(int(mnist.train.num_examples/batch_size)): #dividir 
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
            train,loss=sess2.run([train_step,cross_entropy], feed_dict={X: batch_xs, Ydata: batch_ys})
            #calcular gradiente y pérdida sobre los datos del batch (se usan todas la variables de la gráfica)
            epoch_loss+=loss #calcula la pérdida por cada batch y la suma para tener la pérdida final en toda la época
        
        #print("Epoch:=",epoch+1,"; \t Epoch Loss:=",epoch_loss)
        summary,t_acc=sess2.run([merged,accuracy], feed_dict={X: mnist.train.images, Ydata: mnist.train.labels})#calculates accuracy across all data
        train_writer.add_summary(summary, epoch+1)
        #print("Training Accuracy is", t_acc*100,"%")
        summary,v_acc=sess2.run([merged,accuracy], feed_dict={X: mnist.validation.images,Ydata:mnist.validation.labels})
        valid_writer.add_summary(summary, epoch+1)
        #print("Validation Accuracy is", v_acc*100,"%")
        summary,f_acc=sess2.run([merged,accuracy], feed_dict={X: mnist.test.images,Ydata:mnist.test.labels})
        test_writer.add_summary(summary, epoch+1)
        #print("Test Accuracy is", f_acc*100,"%")
        
        sum_data=np.vstack((sum_data,np.array([epoch+1,epoch_loss,t_acc,v_acc,f_acc])))
        print("Finished Epoch",epoch+1)
