import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import random

training_file="data/order_box_goods_app_filter.txt"
code_file="data/goods_code.txt"

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

def model_feed_data(pos,bs,data_list):
    data_batch=data_list[pos*bs:pos*bs+bs]
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
    return np.array(y),feature_index_batch,feature_values_batch,np.array(data_batch["order_id"])

def get_code():
    i=1
    with open("data/goods_code.txt","r") as file:
        for line in file.readlines():
            i+=1
    return i

embedding_dim_size=100
label_output=16
learning_rate=0.001
training_epochs=6
batch_size=32
feature_max_index=get_code()
unit_size=64
sample_count=1000

if __name__=="__main__":
    print(tf.version)
    data=pd.read_csv(training_file,delimiter="\t",names=["order_id","feature_list","label","warehouse"])
    print(np.shape(data))
    print(data["label"].value_counts())
    test_data = data[data["warehouse"]=="日本仓库"].sample(n=sample_count,random_state=random.randint(0,10000))
    train_data = data.drop(test_data.index)
    # train_data = data
    print(np.shape(train_data))
    print(train_data["label"].value_counts())
    print(np.shape(test_data))
    print(test_data["label"].value_counts())
    train_data.to_csv("data/train.csv",index=False,header=False)
    test_data.to_csv("data/test.csv",index=False,header=False)

    train_data=pd.read_csv("data/train.csv",delimiter=",",names=["order_id","feature_list","label","warehouse"])
    test_data=pd.read_csv("data/test.csv",delimiter=",",names=["order_id","feature_list","label","warehouse"])

    with tf.name_scope("input"):
        feature_index = tf.sparse_placeholder(tf.int32, shape=[None, None], name='feature_index')
        feature_values = tf.sparse_placeholder(tf.float32, shape=[None, None], name='feature_values')
        y_batch = tf.placeholder(tf.int64, shape=[None], name="label")

    with tf.name_scope("embedding"):
        feature_embeddings = tf.get_variable(name='feature_embeddings',
                                             shape=[feature_max_index+1, embedding_dim_size],
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1,seed=2021, dtype=tf.float32),
                                             dtype=tf.float32)

    with tf.name_scope("model"):
        embedding_sum_val=tf.nn.safe_embedding_lookup_sparse(feature_embeddings, feature_index, feature_values, combiner="sum")
        action_layer=tf.layers.dense(embedding_sum_val,embedding_dim_size,activation=tf.nn.relu)
        weights = tf.Variable(tf.truncated_normal([embedding_dim_size, label_output], mean=0.0, stddev=0.1))
        biases = tf.Variable(tf.truncated_normal([label_output], mean=0.0, stddev=0.1))
        output_val = tf.matmul(action_layer, weights) + biases

    with tf.name_scope("optimization"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y_batch,depth=label_output), logits=output_val))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        # logits = tf.nn.softmax(add_val, name="score")
        logits = tf.nn.softmax(output_val,name="score")

    with tf.name_scope("accuracy"):
        logits_index=tf.arg_max(output_val,1)
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
            loop=0
            for i in range(total_batch):
                y,fea_index,fea_vals,order_ids=model_feed_data(i,batch_size,train_data)
                index, values, shape=_sparse_tuple_from(fea_index,feature_max_index)
                fea_vals_index, fea_vals_values, fea_vals_shape=_sparse_tuple_from_val(fea_index,fea_vals,feature_max_index)
                _,loose,acc=sess.run([optimizer,cost,accuracy],feed_dict={y_batch: y, feature_index: tf.SparseTensorValue(index, values, shape),
                                    feature_values: tf.SparseTensorValue(fea_vals_index, fea_vals_values, fea_vals_shape)})
                # print(acc,len(predict),len(tagert),len(order_ids),order_ids,predict,tagert)
                all_losss+=loose
                all_acc+=acc
                loop+=1
                if(loop%500==0):
                    test_y, test_fea_index, test_fea_vals,_ = model_feed_data(0, len(test_data), test_data)
                    t_index, t_values, t_shape = _sparse_tuple_from(test_fea_index, feature_max_index)
                    t_fea_vals_index, t_fea_vals_values, t_fea_vals_shape = _sparse_tuple_from_val(test_fea_index,test_fea_vals,feature_max_index)
                    all_test_acc= sess.run(accuracy, feed_dict={y_batch: test_y, feature_index: tf.SparseTensorValue(t_index, t_values, t_shape),
                                                                                   feature_values: tf.SparseTensorValue(t_fea_vals_index,t_fea_vals_values,t_fea_vals_shape)})
                    print(epoch,i,all_losss/loop,all_acc/loop,all_test_acc)


        # p_list=[]
        # t_list=[]
        # order_ids_list=[]
        # all_acc=0
        # tb=int(len(train_data)/10000)+1
        # for j in range(tb):
        #     y, fea_index, fea_vals,order_ids = model_feed_data(j,10000,train_data)
        #     index, values, shape = _sparse_tuple_from(fea_index, feature_max_index)
        #     fea_vals_index, fea_vals_values, fea_vals_shape = _sparse_tuple_from_val(fea_index, fea_vals,feature_max_index)
        #     acc,predict,tagert = sess.run([accuracy,logits_index,y_batch],feed_dict={y_batch: y, feature_index: tf.SparseTensorValue(index, values, shape),
        #                                         feature_values: tf.SparseTensorValue(fea_vals_index, fea_vals_values,fea_vals_shape)})
        #     all_acc += acc
        #     p_list.extend(predict)
        #     t_list.extend(tagert)
        #     order_ids_list.extend(order_ids)
        #     print(len(p_list),len(t_list),len(order_ids_list))
        # print("train_acc:",all_acc/tb)

        # train_predict_file = open("data/train_boxs_predict.txt", "w", encoding="utf-8")
        # for o, p, t in zip(order_ids_list, p_list, t_list):
        #     # print(f,p,t)
        #     train_predict_file.write("%s\t%s\t%s\n" % (str(o), str(p), str(t)))
        # train_predict_file.close()


        test_y, test_fea_index, test_fea_vals,_ = model_feed_data(0, len(test_data), test_data)
        t_index, t_values, t_shape = _sparse_tuple_from(test_fea_index, feature_max_index)
        t_fea_vals_index, t_fea_vals_values, t_fea_vals_shape = _sparse_tuple_from_val(test_fea_index, test_fea_vals, feature_max_index)
        predict,tagert = sess.run([logits_index,y_batch], feed_dict={y_batch: test_y, feature_index: tf.SparseTensorValue(t_index, t_values, t_shape),
                                           feature_values: tf.SparseTensorValue(t_fea_vals_index, t_fea_vals_values,t_fea_vals_shape)})

        order_id_list=test_data["order_id"]
        fea_file=open("data/test_boxs_predict.txt","w",encoding="utf-8")
        for o,p,t in zip(order_id_list,predict,tagert):
            # print(f,p,t)
            fea_file.write("%s\t%s\t%s\n"%(str(o),str(p),str(t)))
        fea_file.close()

        builder = tf.saved_model.builder.SavedModelBuilder("./boxs_model/")
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING]
        )
        builder.save()
