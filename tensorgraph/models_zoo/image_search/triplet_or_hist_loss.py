
import tensorflow as tf

def triplet_loss(labels, embeddings, alpha, target, labels_size, target_size, penalize_ratio, squared=True, epsilon=1e-8, name = 'batch_all_triplet_loss'):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    with tf.variable_scope(name):
        # Get the pairwise distance matrix
        pairwise_dist = _pairwise_distances(embeddings, squared=True)

        ap_mask = pos_penalize_mask(labels, target, labels_size, target_size)
        an_mask = neg_penalize_mask(labels, target, labels_size, target_size)
        anchor_positive_dist = tf.where(ap_mask, pairwise_dist + penalize_ratio, pairwise_dist)
        anchor_negative_dist = tf.where(an_mask, pairwise_dist - penalize_ratio, pairwise_dist)

#        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)# shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(anchor_positive_dist, 2)# shape (batch_size, batch_size, 1)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)

#        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)# shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(anchor_negative_dist, 1)# shape (batch_size, 1, batch_size)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + alpha

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(labels)
        mask = tf.to_float(mask)
        triplet_loss = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def pos_penalize_mask(labels, target, labels_size, target_size):
    """
    args:
        labels: tensor list of labels
        target: tensor list of target labels
        labels_size: integer
        target_size: integer
    return:
        mask to determine which positive pairwise distance to penalize
    """
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    indices_equal = tf.cast(tf.eye(labels_size), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    exist_in_target = tf.tile(tf.expand_dims(labels, axis = 1), [1, target_size])
    exist_in_target = tf.equal(exist_in_target, target)
    exist_in_target = tf.cast(exist_in_target, dtype = tf.int32)
    exist_in_target = tf.reduce_sum(exist_in_target, axis = 1)
    exist_in_target = tf.cast(exist_in_target, dtype = tf.bool)
    exist_in_target = tf.logical_and(tf.expand_dims(exist_in_target, axis=0), tf.expand_dims(exist_in_target, axis=1))
    return tf.logical_and(tf.logical_and(exist_in_target, labels_equal), indices_not_equal)

def neg_penalize_mask(labels, target, labels_size, target_size):
    """
    args:
        labels: tensor list of labels
        target: tensor list of target labels
        labels_size: integer
        target_size: integer
    return:
        mask to determine which negative pairwise distance to penalize
    """
    labels_not_equal = tf.not_equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    indices_equal = tf.cast(tf.eye(labels_size), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    exist_in_target = tf.tile(tf.expand_dims(labels, axis = 1), [1, target_size])
    exist_in_target = tf.equal(exist_in_target, target)
    exist_in_target = tf.cast(exist_in_target, dtype = tf.int32)
    exist_in_target = tf.reduce_sum(exist_in_target, axis = 1)
    exist_in_target = tf.cast(exist_in_target, dtype = tf.bool)
    exist_in_target = tf.logical_or(tf.expand_dims(exist_in_target, axis=0), tf.expand_dims(exist_in_target, axis=1))
    return tf.logical_and(tf.logical_and(exist_in_target, labels_not_equal), indices_not_equal)


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask

def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_not(labels_equal)
    return mask

def positive_penalize_fn(positive_similarity, penalize_ratio):
    return tf.maximum(tf.subtract(positive_similarity, penalize_ratio), -1.0)

def negative_penalize_fn(negative_similarity, penalize_ratio):
    return tf.minimum(tf.add(negative_similarity, penalize_ratio), 1.0)

def nth(tensor):
    return tensor


def histogram_loss(labels, embeddings, target, labels_size, target_size, penalize_ratio, name = 'batch_all_histogram_loss'):
     """Build the histogram loss over a batch of embeddings.
     Args:
         labels: labels of the batch, of size (batch_size,)
         embeddings: tensor of shape (batch_size, embed_dim)
     Returns:
         histogram_loss: scalar tensor containing the histogram loss
     """
     with tf.variable_scope(name):
         dim = embeddings.shape[1]
         R = tf.constant(dim, tf.int32)

         # Get the pairwise cosine similarity matrix
         pairwise_similarity = tf.matmul(embeddings, embeddings, transpose_b = True) # (batchsize, batchsize) matrix with pairwise similarity

         positive_mask = tf.to_float(_get_anchor_positive_triplet_mask(labels)) #(batchsize, batchsize) matrix with 1's at valid positive pairs indices
         negative_mask = tf.to_float(_get_anchor_negative_triplet_mask(labels)) #(batchsize, batchsize) matrix with 1's at valid negative pairs indices

         positive_similarity = tf.multiply(positive_mask, pairwise_similarity)
         positive_similarity = tf.where(pos_penalize_mask(labels, target, labels_size, target_size),
                                         positive_penalize_fn(positive_similarity, penalize_ratio),
                                         nth(positive_similarity))
         lower_positive_similarity = tf.matrix_band_part(positive_similarity, -1, 0) #lower triangular (batchsize, batchsize) matrix with positive pair's pairwise similarity
         flat_positive_similarity = tf.gather_nd(lower_positive_similarity, tf.where(tf.not_equal(lower_positive_similarity, 0.0))) #flatten the matrix

         negative_similarity = tf.multiply(negative_mask, pairwise_similarity)
         negative_similarity = tf.where(neg_penalize_mask(labels, target, labels_size, target_size),
                                         negative_penalize_fn(negative_similarity, penalize_ratio),
                                         nth(negative_similarity))
         lower_negative_similarity = tf.matrix_band_part(negative_similarity, -1, 0) #lower triangular (batchsize, batchsize) matrix with negative pair's pairwise similarity
         flat_negative_similarity = tf.gather_nd(lower_negative_similarity, tf.where(tf.not_equal(lower_negative_similarity, 0.0))) #flatten the matrix

         nbr_pos_bins = tf.Variable(dim, dtype=tf.int32)
         nbr_neg_bins = tf.Variable(dim, dtype=tf.int32)

         flat_positive_similarity = tf.multiply(flat_positive_similarity, tf.divide(tf.to_float(nbr_pos_bins), 2.0))
         flat_negative_similarity = tf.multiply(flat_negative_similarity, tf.divide(tf.to_float(nbr_neg_bins), 2.0))

         sorted_flat_positive_similarity = tf.contrib.framework.sort(flat_positive_similarity)
         sorted_flat_negative_similarity = tf.contrib.framework.sort(flat_negative_similarity)

         floor_pos_pos = tf.map_fn(lambda x: tf.floor(x), sorted_flat_positive_similarity, dtype = tf.float32)
         floor_pos_value = tf.map_fn(lambda x: tf.subtract(tf.ceil(x), x), sorted_flat_positive_similarity, dtype=tf.float32)
         ceil_pos_pos = tf.map_fn(lambda x: tf.ceil(x), sorted_flat_positive_similarity, dtype = tf.float32)
         ceil_pos_value = tf.map_fn(lambda x: tf.subtract(x, tf.floor(x)), sorted_flat_positive_similarity, dtype=tf.float32)

         floor_neg_pos = tf.map_fn(lambda x: tf.floor(x), sorted_flat_negative_similarity, dtype = tf.float32)
         floor_neg_value = tf.map_fn(lambda x: tf.subtract(tf.ceil(x), x), sorted_flat_negative_similarity, dtype=tf.float32)
         ceil_neg_pos = tf.map_fn(lambda x: tf.ceil(x), sorted_flat_negative_similarity, dtype = tf.float32)
         ceil_neg_value = tf.map_fn(lambda x: tf.subtract(x, tf.floor(x)), sorted_flat_negative_similarity, dtype=tf.float32)

         multiples = [dim,1]

         compare = tf.range(-R/2, R/2, 1) ###
         compare = tf.expand_dims(compare,axis=-1)

         floor_pos_pos = tf.expand_dims(floor_pos_pos, axis = 0)
         floor_pos_pos = tf.tile(floor_pos_pos, multiples)
         temp1 = tf.cast(tf.equal(floor_pos_pos, tf.to_float(compare)), dtype = tf.float32)
         floor_pos_hist = tf.matmul(temp1, tf.expand_dims(floor_pos_value, axis=0), transpose_b = True)

         ceil_pos_pos = tf.expand_dims(ceil_pos_pos, axis = 0)
         ceil_pos_pos = tf.tile(ceil_pos_pos, multiples)
         temp2 = tf.cast(tf.equal(ceil_pos_pos, tf.to_float(compare)), dtype = tf.float32)
         ceil_pos_hist = tf.matmul(temp2, tf.expand_dims(ceil_pos_value, axis=0), transpose_b = True)

         total_pos_hist = tf.add(floor_pos_hist, ceil_pos_hist)
         total_pos_hist = tf.divide(total_pos_hist, tf.divide(tf.reduce_sum(positive_mask), 2.0))

         floor_neg_pos = tf.expand_dims(floor_neg_pos, axis = 0)
         floor_neg_pos = tf.tile(floor_neg_pos, multiples)
         temp3 = tf.cast(tf.equal(floor_neg_pos, tf.to_float(compare)), dtype = tf.float32)
         floor_neg_hist = tf.matmul(temp3, tf.expand_dims(floor_neg_value, axis=0), transpose_b = True)

         ceil_neg_pos = tf.expand_dims(ceil_neg_pos, axis = 0)
         ceil_neg_pos = tf.tile(ceil_neg_pos, multiples)
         temp4 = tf.cast(tf.equal(ceil_neg_pos, tf.to_float(compare)), dtype = tf.float32)
         ceil_neg_hist = tf.matmul(temp4, tf.expand_dims(ceil_neg_value, axis=0), transpose_b = True)

         total_neg_hist = tf.add(floor_neg_hist, ceil_neg_hist)
         total_neg_hist = tf.divide(total_neg_hist, tf.divide(tf.reduce_sum(negative_mask), 2.0))

         cum_total_pos_hist = tf.cumsum(total_pos_hist)
         hist_loss = tf.multiply(total_neg_hist, cum_total_pos_hist)
         total_hist_loss = tf.reduce_sum(hist_loss)

     return total_hist_loss


