---
layout: post
title: Effective training of partial models in tensorflow!
---

Let's say you want to train a model in tensorflow consisting of multiple sub-models and for some reason
you cannot connect their computation graphs directly. 

As an example, consider the tensorflow translation model (read [here](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html) before going on). 
It consists of an encoder and a decoder RNN. Input sequences to both encoder and decoder can vary,
so we need to create buckets for different encoder-/decoder-sequence-size pairs. Fortunately,
for the translation model the encoder- and decoder-sequences have similar sizes, i.e., they are in 
general not independent from each other. Large encoder-sequences come with large decoder-sequences, and 
small encoder-sequences come with small decoder-sequences. Thus, buckets of about equal encoder- 
to decoder-sizes can be created.

Imagine, however, this was not the case, i.e., encoder- and decoder-sequence lengths are completely independent. 
We would have to create many more buckets to stay efficient, e.g., also large to small and small to large buckets. 
So the number of created computation-graphs would grow dramatically which would not be memory-efficient. If we incorporate even more RNNs (e.g., a second encoder) into the overall model this becomes even worse.

One solution would be to run individual models seperately and carry over outputs (and gradients for backpropagation)
, respectively. Staying with the translation example, this means, that we run the encoder to receive its 
outputs, and carry these outputs over to the decoder as input. We run the decoder, do backpropagation and receive
a gradient on the encoder output. Then we would like to run only backpropagation on the encoder model given the computed
gradient on its output, without having to make a forward pass again. This, however, turns out to be non-trivial 
(at least with my current understanding of tensorflow). 

The following code creates a simple computation graph:

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
# upstream gradient on output d, that is manually set by a driver program
grad_d = tf.placeholder(tf.float32)
# adding gradient ops to computation graph w.r.t to given upstream gradient
grad_a = tf.gradients(d,[a], grad_d) 
```

Here is the naive way of splitting the forward and backward pass:

```python
out = s.run(d)
# --> logger will show "I am running forward!"

# use out to compute grad_d, e.g., with another model
# ...

# backprop
grad_a_result = s.run(grad_a, feed_dict={grad_d: 0.1})
# --> logger will show "I am running forward!", the second time forward ran

```

The naive implementation results in running forward twice, which is not efficient, because the backward pass is tied
to nearly all operations of the forward-pass. 
In order to only run the backward-pass (backpropagation), we have to make use of Tensorflow's partial computation ability. It is able to compute only partial computation-graphs depending on the provided input and expected output.
Therefore, the only way to not compute anything twice is to store the state of all 
encoder-forward-operations after its forward pass, which requires us to traverse the computation-graph from
the encoder output backward until its input to collect all necessary operations and fetch their results during the forward pass.

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
# --> logger will show "I am running forward!"

# create input feed
feed_dict = {grad_d: 0.1}
for i in xrange(len(comp_tensors)):
    feed_dict[dep_tensors[i]] = comp_tensors[i]
    
    
out = feed_dict[d]

# backward pass

grad_a_result = s.run(grad_a, feed_dict = feed_dict)
# --> logger will NOT show "I am running forward!"
```

Now, this solution seems somewhat like a hack, because the user of tensorflow has to traverse 
the computation graph manually. However, I could not find a better way of
dealing with this problem. Any questions, suggestions or better solutions, please leave
a comment or write an email?
