# MLSMOTE
This is a test implementation for the python version of multi-label imbalanced learning algorithm MLSMOTE.

```python
data = np.random.rand(100, 100, 5)
labels = np.random.randint(15, size=(100, 15))
n = 3
k = 10
tail_list = [1, 4, 6, 8]
model = Mlsmote(data, labels, tail_list, n, k)
data_wrap, labels_wrap = model.MlS()

print(data_wrap.shape, labels_wrap.shape)
```


And the output is like this:

```
self.N is 3
time total knn 0.046033382415771484
(1252, 100, 5) (1252, 15)
```
