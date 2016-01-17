---
layout: post
title: Effective training of partial models in tensorflow!
---

Let's say you want to train a model in tensorflow consisting of multiple sub-models and for some reason
you cannot connect their computation graphs directly. 

As an example, consider the tensorflow translation model (read [here](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html) before going on). 
It consists of an encoder and a decoder RNN. Input sequences to both encoder and decoder can vary,
so we need to create buckets for different encoder-decoder size pairs of these sequences. Fortunately,
for the translation model the encoder- and decoder-sequences have similar sizes, i.e., they are in 
general not independent from each other. Large encoder-sequences come with large decoder-sequences, and 
small encoder-sequences come with small decoder-sequences. Thus, buckets of about equal encoder- 
to decoder-sizes can be created.

Imagine, however, this was not the case, i.e., encoder- and decoder-sequence lengths are completely independent. 
We would need to create many more buckets stay efficient, e.g., also large to small and small to large buckets. 
So the number of created models would grow dramatically which would not be memory-efficient. If we incorporate even
more RNNs (e.g., a second encoder) into the overall model this becomes even worse.

One solution would be to run individual models seperately and carry over outputs (and gradients for backpropagation)
, respectively. Staying with the translation example, this means, that we run the encoder to receive its 
outputs, and carry these outputs over to the decoder as input. We run the decoder, do backpropagation and receive
a gradient on the encoder output. Then we would like to run only backpropagation on the encoder model given the computed
gradient on its output, without having to make a forward pass again. This, however, turns out to be non-trivial 
(at least with my current understanding of tensorflow). 

In order to do only do backpropagation, we have to make use of Tensorflow's partial computation ability.
It is able to compute only partial computation-graphs depending on the provided input and expected output.
Therefore, the only way to not compute anything twice, we would need to store the state of all 
encoder-forward-operations after its forward pass, which requires us to traverse the computation-graph from
the encoder output backward until its input. 

The following code exemplifies the retrieval of the necessary state information for a simple computation graph,
and how to use it to split the forward- and backward pass (backpropagation).

```python
# Setup
import tensorflow as tf
s = tf.InteractiveSession()

# Model definition
a = tf.Variable(1.0)
b = a + a
# Add a print operation to see whether there was a forward pass
c = tf.Print(b,[b],message="I am running forward!")
d = c + c

# Add gradient computation ops to computation graph
grad_from_model2 = tf.placeholder(tf.float32)
grad = tf.gradients(d,[a], grad_from_model2)
```

Here is the naive way of doing forward and backward with intermediate call to another model (or more):

```python
out = s.run(d)
# --> logger will show "I am running forward!"

# use out as input for a second model to receive grad_from_model2
# ...

# backprop
grad_a = s.run(grad, feed_dict={grad_from_model2: 0.1})
# --> logger will show "I am running forward!", the second time forward ran

```

Efficient solution:

```python
# first compute all participating tensors in forward pass
def get_tensors(output_tensors, input_tensors):
    res = set()
    for o in output_tensors:
        if o not in input_tensors: # we do not want to add placeholders, for example
            res.add(o)
            res = res.union(get_tensors(o.op.inputs,input_tensors))
    return res

# tensors d depends on
dep_tensors = list(get_tensors([d],[]))

# forward pass
comp_tensors = s.run(dep_tensors)

# create input feed
feed_dict = {grad_from_model2: 0.1}
for i in xrange(len(comp_tensors)):
    feed_dict[dep_tensors[i]] = comp_tensors[i]
    
    
out = feed_dict[d]
# --> logger will show "I am running forward!"

# backward pass

grad_a = s.run(grad, feed_dict = feed_dict)
# --> logger will NOT show "I am running forward!"
```


Now, this solution seems somewhat like a hack, but I could not find a better way of
addressing this problem. Any questions, suggestions or better solutions, please leave
a comment?
