import numpy as np
import torch
import tensorflow as tf
import tf_geometric as tfg
import tensorflow_probability as tfp


from tf_geometric.utils.graph_utils import negative_sampling
from tqdm import tqdm
from sklearn.cluster import KMeans

from models.owgl_modules import MultiVariationalGCNWithDense
from utils import reassign_labels, reverse_labels

class OpenWGL():
    def __init__(self, data, expected_num_classes , seed, learning_rate = 1e-3, n_epochs=30):
        self.graph = tfg.Graph(
                            x=data.x.detach().numpy(),  # 5 nodes, 20 features,
                            edge_index= data.edge_index.detach().numpy(),  # 4 undirected edges
                            y = data.y.detach().numpy())
        
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.unseen_label_index = -1
        self.expected_num_classes = expected_num_classes
        
        self.original_num_classes = np.max(self.graph.y) + 1
        self.seen_labels = data.known_classes.tolist()
        self.unknown_classes = data.unknown_classes.numpy()
        self.y_true = reassign_labels(self.graph.y, self.seen_labels, self.unseen_label_index)
        self.train_indices = torch.nonzero(data.labeled_mask).squeeze().detach().numpy()
        self.valid_indices = torch.nonzero(data.all_class_val_mask).squeeze().detach().numpy()
        self.test_indices = torch.nonzero(data.all_class_val_mask).squeeze().detach().numpy()
        self.num_classes = np.max(self.y_true) + 1
        
        self.use_softmax = True
        self.use_class_uncertainty = True
        self.use_VGAE = True
        self.uncertain_num_samplings = 100 if self.use_VGAE else 1
        self.seed = seed
        
        self.model = MultiVariationalGCNWithDense([32, 16, self.num_classes],
                                    uncertain=self.use_VGAE,
                                    output_list=True)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    def logits_to_probs(self, logits):
        if self.use_softmax:
            probs = tf.nn.softmax(logits)
        else:
            probs = tf.nn.sigmoid(logits)
        return probs

    def get_threshold(self, logits, mask_indices):

        if isinstance(logits, list):
            logits_list = tf.stack(logits, axis=-1)
            logits = tf.reduce_mean(logits_list, axis=-1)
    
            if self.use_softmax:
                probs_list = tf.nn.softmax(logits_list, axis=-2)
            else:
                probs_list = tf.nn.sigmoid(logits_list)
            probs = tf.reduce_mean(probs_list, axis=-1)
        else:
            probs = self.logits_to_probs(logits)
    
        masked_logits = tf.gather(logits, mask_indices)
        masked_y_pred = tf.argmax(masked_logits, axis=-1)
        masked_y_true = self.y_true[mask_indices]
    
        probs = tf.gather(probs, mask_indices)
        probs = tf.gather_nd(probs, tf.stack([tf.range(masked_logits.shape[0], dtype=tf.int64), masked_y_pred], axis=1))
        probs = probs.numpy()
        
        threshold = (probs[masked_y_true != self.unseen_label_index].mean()+probs[masked_y_true == self.unseen_label_index].mean()) / 2.0
    
        return threshold

    def get_predictions(self, logits, mask_indices, threshold):

        if isinstance(logits, list):
            logits_list = tf.stack(logits, axis=-1)
            logits = tf.reduce_mean(logits_list, axis=-1)
    
            if self.use_softmax:
                probs_list = tf.nn.softmax(logits_list, axis=-2)
            else:
                probs_list = tf.nn.sigmoid(logits_list)
            probs = tf.reduce_mean(probs_list, axis=-1)
        else:
            probs = self.logits_to_probs(logits)
    
        masked_logits = tf.gather(logits, mask_indices)
        masked_y_pred = tf.argmax(masked_logits, axis=-1)
        masked_y_true = self.y_true[mask_indices]
    
        
        probs = tf.gather(probs, mask_indices)
        probs = tf.gather_nd(probs, tf.stack([tf.range(masked_logits.shape[0], dtype=tf.int64), masked_y_pred], axis=1))
        probs = probs.numpy()
        masked_y_pred = masked_y_pred.numpy()
        masked_y_pred[probs < threshold] = self.unseen_label_index
    
        return masked_y_pred


    def compute_loss(self, outputs, kl, mask_indices):
        # use negative_sampling
        logits = outputs[-1]
        h = outputs[-2]
    
        if self.use_VGAE:
            neg_edge_index = negative_sampling(
                num_samples=self.graph.num_edges,
                num_nodes=self.graph.num_nodes,
                edge_index=None,
                replace=False
            )
    
            pos_logits = tf.reduce_sum(
                tf.gather(h, self.graph.edge_index[0]) * tf.gather(h, self.graph.edge_index[1]),
                axis=-1
            )
            neg_logits = tf.reduce_sum(
                tf.gather(h, neg_edge_index[0])  * tf.gather(h, neg_edge_index[1]),
                axis=-1
            )
    
            pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos_logits,
                labels=tf.ones_like(pos_logits)
            )
            neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg_logits,
                labels=tf.zeros_like(neg_logits)
            )
            gae_loss = tf.reduce_mean(pos_losses) + tf.reduce_mean(neg_losses)
    
    
        all_indices = np.arange(0, tf.shape(logits)[0])
        unmasked_indices = np.delete(all_indices, mask_indices)
    
        unmasked_logits = tf.gather(logits, unmasked_indices)
        #
        loss_func = tf.nn.softmax_cross_entropy_with_logits if self.use_softmax else tf.nn.sigmoid_cross_entropy_with_logits
    
        unmasked_probs = self.logits_to_probs(unmasked_logits)
        unmasked_probs = tf.clip_by_value(unmasked_probs, 1e-7, 1.0)

        #is_empty = tf.equal(tf.size(unmasked_probs), 0)
       
        unmasked_preds = tf.argmax(unmasked_probs, axis=-1)
        unmasked_prob = tf.gather_nd(unmasked_probs, tf.stack([tf.range(unmasked_logits.shape[0], dtype=tf.int64), unmasked_preds], axis=1))
            
    
    
        upper_bound = tfp.stats.percentile(unmasked_prob, q=90.)
    
        topk_indices = tf.where(tf.logical_and(
            tf.greater(unmasked_prob, 1.0 / self.num_classes),
            tf.less(unmasked_prob, upper_bound)
        ))
        
    
        unmasked_probs = tf.gather(unmasked_probs, topk_indices)
        class_uncertainty_losses = unmasked_probs * tf.math.log(unmasked_probs)
    
        masked_logits = tf.gather(logits, mask_indices)
        masked_y_true = self.y_true[mask_indices]
        losses = loss_func(
            logits=masked_logits,
            labels=tf.one_hot(masked_y_true, depth=self.num_classes)
        )
        masked_kl = tf.gather(kl, mask_indices)
    
        loss = tf.reduce_mean(losses)
    
    
        if self.use_class_uncertainty:
            loss += tf.reduce_mean(class_uncertainty_losses) * 1.0
    
    
        if self.use_VGAE:
            
            loss = loss + gae_loss * 1.0 + tf.reduce_mean(masked_kl)*1.0
    
        return loss

    def train_model(self):
        
        for step in tqdm(range(self.n_epochs)):
        
            with tf.GradientTape() as tape:
                outputs, kl = self.model([self.graph.x, self.graph.edge_index, self.graph.edge_weight], cache=self.graph.cache, training=True)
                logits = outputs[-1]
                train_loss = self.compute_loss(outputs, kl, self.train_indices)
        
            vars = tape.watched_variables()
            grads = tape.gradient(train_loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

    def get_embeddings_and_labels(self):
        outputs, kl = self.model([self.graph.x, self.graph.edge_index, self.graph.edge_weight], cache=self.graph.cache, training=True)
        logits = outputs[-1]
        threshold = self.get_threshold(logits, self.valid_indices)
        
        y_test = self.get_predictions(logits,  tf.range(0,  self.graph.x.shape[0]), threshold)
        y_open = reverse_labels(y_test, self.seen_labels, self.unseen_label_index)

        embeddings = self.model.get_embeddings([self.graph.x, self.graph.edge_index, self.graph.edge_weight], cache=self.graph.cache, training=False)
        embeddings = embeddings.numpy()

        return embeddings, y_open

    def get_open_predictions(self):
        
        embeddings, y_open = self.get_embeddings_and_labels()
        novel_class_embeddings = embeddings[y_open==self.unseen_label_index,:]
        
        if novel_class_embeddings.shape[0] == 0:
            return y_open
            
        else:
            k = self.expected_num_classes - self.num_classes
            
            kmeans = KMeans(n_clusters = k, random_state=self.seed, n_init="auto").fit(novel_class_embeddings)
            open_labels = self.unknown_classes[kmeans.labels_]
            y_final = y_open
            y_final[y_open==self.unseen_label_index] = open_labels
            
            return y_final







