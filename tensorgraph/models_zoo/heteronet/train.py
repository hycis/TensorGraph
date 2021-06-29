import tensorflow as tf
from ...cost import entropy

class HeteroTrain():
    def __init__(self,model, t1_ph, t2_ph, tc_ph, y_ph):
        y_train_sb    = model.train_fprop(t1_ph,t2_ph,tc_ph)[0]
        train_cost_sb = entropy(y_ph, y_train_sb)
        optimizer     = tf.train.AdamOptimizer(0.0001)
        opt = optimizer.minimize(train_cost_sb)
        self.sess =  tf.Session()
        print('initialize global_variables')
        self.sess.run(tf.global_variables_initializer())
        self.model         = model
        self.train_cost_sb = train_cost_sb
        self.optimizer     = opt
        
    def train(self,feed_dict):
        cost, _   = self.sess.run([self.train_cost_sb,self.optimizer],feed_dict=feed_dict)
        print(cost)
        
