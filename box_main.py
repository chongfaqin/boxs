import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

training_file="data/order_box_feature_app_filter.txt"
# test_file="data/order_box_feature.txt"
embedding_dim_size=128
label_output=16
learning_rate=0.001
training_epochs=3
batch_size=32
feature_max_index=63624
test_batch_size=320
def _sparse_tuple_from(src_seq_list,nwords):
    """
    convert a list to sparse tuple
    usage:
        src_seq_list = [[1,2],[1],[1,2,3,4,5],[2,5,2,6]]
        sparse_tensor = sparse_tuple_from(src_seq_list)
        then :
            sparse_tensor[0](indices) = [[0,0],[0,1],[1,0],[2,0],[2,1],[2,2],[2,3],[2,4],[3,0],[3,1],[3,2],[3,3]]
            sparse_tensor[1](values) =  [1,2,1,1,2,3,4,5,2,5,2,6], squeezed src_seq_list's values
            sparse_tensor[2](shape) =  [4,5] , 4: number of sequence; 5: max_length of seq_labels
    """
    indices = []
    values = []
    for n, seq in enumerate(src_seq_list):
        if(len(seq)==0):
            continue
        indices.extend(zip([n] * np.shape(seq)[0], seq))
        values.extend(seq)

    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    shape = np.array([np.shape(src_seq_list)[0], nwords+1], dtype=np.int64)
    return indices, values, shape

def _sparse_tuple_from_val(src_idx_list,src_val_list,nwords):
    """
    convert a list to sparse tuple
    usage:
        src_seq_list = [[1,2],[1],[1,2,3,4,5],[2,5,2,6]]
        sparse_tensor = sparse_tuple_from(src_seq_list)
        then :
            sparse_tensor[0](indices) = [[0,0],[0,1],[1,0],[2,0],[2,1],[2,2],[2,3],[2,4],[3,0],[3,1],[3,2],[3,3]]
            sparse_tensor[1](values) =  [1,2,1,1,2,3,4,5,2,5,2,6], squeezed src_seq_list's values
            sparse_tensor[2](shape) =  [4,5] , 4: number of sequence; 5: max_length of seq_labels
    """
    # print(len(src_seq_list))
    indices = []
    values = []
    for n, seq in enumerate(src_idx_list):
        if(len(seq)==0):
            continue
        indices.extend(zip([n] * np.shape(seq)[0], seq))
        values.extend(src_val_list[n])

    indices = np.array(indices, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    shape = np.array([np.shape(src_idx_list)[0], nwords+1], dtype=np.int64)
    return indices, values, shape

def model_feed_data(pos,batch_size,data_list):
    data_batch=data_list[pos:pos+batch_size]
    y=data_batch["label"]
    feature_list=data_batch["feature_list"]
    feature_index_batch=[]
    feature_values_batch=[]
    for line in feature_list:
        try:
            features=line.strip().split(",")
            fea_index=[]
            fea_values=[]
            for fea in features:
                fea_arr=fea.split(":")
                fea_index.append(fea_arr[0])
                fea_values.append(fea_arr[1])
            feature_index_batch.append(fea_index)
            feature_values_batch.append(fea_values)
        except Exception as e:
            print(e,line)
    # print(np.shape(feature_index_batch),np.shape(feature_values_batch))
    return np.array(y),feature_index_batch,feature_values_batch

if __name__=="__main__":
    print(tf.version)
    data=pd.read_csv(training_file,delimiter="\t",names=["feature_list","label"])
    print(np.shape(data))
    print(data["label"].value_counts())
    test_data = data.sample(n=30000)
    train_data = data.drop(test_data.index)
    print(np.shape(train_data))
    print(train_data["label"].value_counts())
    print(np.shape(test_data))
    print(test_data["label"].value_counts())
    train_data.to_csv("data/train.csv",index=False)
    test_data.to_csv("data/test.csv",index=False)

    with tf.name_scope("input"):
        feature_index = tf.sparse_placeholder(tf.int32, shape=[None, None], name='feature_index')
        feature_values = tf.sparse_placeholder(tf.float32, shape=[None, None], name='feature_values')
        y_batch = tf.placeholder(tf.int64, shape=[None], name="label")

    with tf.name_scope("embedding"):
        feature_embeddings = tf.get_variable(name='feature_embeddings',
                                             shape=[feature_max_index+1, embedding_dim_size],
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.33,seed=None, dtype=tf.float32),
                                             dtype=tf.float32)

    with tf.name_scope("model"):
        embedding_sum_val=tf.nn.embedding_lookup_sparse(feature_embeddings, feature_index, feature_values, combiner="sum")
        weights = tf.Variable(tf.truncated_normal([embedding_dim_size, label_output], mean=0.0, stddev=0.33))
        biases = tf.Variable(tf.truncated_normal([label_output], mean=0.0, stddev=0.33))
        output_val = tf.matmul(embedding_sum_val, weights) + biases

    with tf.name_scope("optimization"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y_batch,depth=label_output), logits=output_val))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        # logits = tf.nn.softmax(add_val, name="score")
        logits = tf.nn.softmax(output_val,name="score")

    with tf.name_scope("accuracy"):
        logits_index=tf.arg_max(logits,1)
        correct_pred = tf.equal(logits_index, y_batch)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # with tf.name_scope("auc"):
        # correct_pred = tf.equal(tf.arg_max(logits, 1), y_batch)
        # auc = tf.metrics.auc(labels=y_batch, predictions=tf.argmax(logits, 1))
        # auc = tf.metrics.auc(labels=tf.reshape(y_batch,[-1,1]), predictions=tf.round(logits))

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()  # log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        total_batch=int(len(train_data)/batch_size)
        for epoch in range(training_epochs):

            train_data = shuffle(train_data)

            all_losss=0
            all_acc=0
            for i in range(total_batch):
                y,fea_index,fea_vals=model_feed_data(i,batch_size,train_data)
                index, values, shape=_sparse_tuple_from(fea_index,feature_max_index)
                fea_vals_index, fea_vals_values, fea_vals_shape=_sparse_tuple_from_val(fea_index,fea_vals,feature_max_index)
                # print(np.shape(index),np.shape(values),np.shape(fea_vals_index),np.shape(fea_vals_values))
                _,loose,acc=sess.run([optimizer,cost,accuracy],feed_dict={y_batch: y, feature_index: tf.SparseTensorValue(index, values, shape),
                                    feature_values: tf.SparseTensorValue(fea_vals_index, fea_vals_values, fea_vals_shape)})
                all_losss+=loose
                all_acc+=acc
                if(i%500==0 and i!=0):
                    y, fea_index, fea_vals = model_feed_data(0, 30000, test_data)
                    index, values, shape = _sparse_tuple_from(fea_index, feature_max_index)
                    fea_vals_index, fea_vals_values, fea_vals_shape = _sparse_tuple_from_val(fea_index, fea_vals,feature_max_index)
                    all_test_acc = sess.run(accuracy,feed_dict={y_batch: y, feature_index: tf.SparseTensorValue(index, values, shape),feature_values: tf.SparseTensorValue(fea_vals_index, fea_vals_values,fea_vals_shape)})
                    print(epoch,i,all_losss/i,all_acc/i,"test_acc:",all_test_acc)

        y, fea_index, fea_vals = model_feed_data(0, 30000, test_data)
        index, values, shape = _sparse_tuple_from(fea_index, feature_max_index)
        fea_vals_index, fea_vals_values, fea_vals_shape = _sparse_tuple_from_val(fea_index, fea_vals, feature_max_index)
        predict,tagert = sess.run([logits_index,y_batch], feed_dict={y_batch: y, feature_index: tf.SparseTensorValue(index, values, shape),
                                           feature_values: tf.SparseTensorValue(fea_vals_index, fea_vals_values,
                                                                                fea_vals_shape)})

        fea_file=open("data/fea_predict.txt","w",encoding="utf-8")
        for f,v,p,t in zip(fea_index,fea_vals,predict,tagert):
            # print(f,p,t)
            fea_file.write("%s\t%s\t%s\t%s\n"%(f,v,str(p),str(t)))
        fea_file.close()
