# MLSMOTE
This is a test implementation for the python version of multi-label imbalanced learning algorithm Multi label Synthetic Minority Over-sampling Technique (MLSMOTE).

Here is a very simple using example for the code.

```python
data = np.random.rand(100, 10, 5)
labels = np.random.randint(2, size=(100, 15))
n = 3
k = 5
model = Mlsmote(data, labels, n, k)
data_wrap, labels_wrap = model.MlS()

print(data_wrap.shape, labels_wrap.shape)

print(data_wrap.shape, labels_wrap.shape)
```


And the output is like this:

```
self.N is 3
time total knn 0.007537364959716797
(1030, 10, 5) (1030, 15)
```
