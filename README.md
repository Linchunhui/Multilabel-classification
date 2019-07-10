# Multilabel-classification
Multilabel classification with mobilenet and inference in ncnn.

# How to do multilabel classification compared to multiclass.
## change label map
If we just want to do multiclass classification,the label map wil be:
> Cat:0
> Dog:1
> Other:2

When we change label map:
> Cat:[1,0,0]
> Dog:[0,1,0]
> Other:[0,0,1]
> Cat_Dog:[1,1,0]
in other words,we comelete operation like `one-hot-encode`.

