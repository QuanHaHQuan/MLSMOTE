# MLSMOTE
This is a test implementation for the python version of multi-label imbalanced learning algorithm Multi label Synthetic Minority Over-sampling Technique (MLSMOTE).

Here is a very simple using example for the code.

```python
data = np.random.rand(100, 10, 5)
labels = np.random.randint(2, size=(100, 8))
n = 3
k = 5
model = Mlsmote(data, labels, n, k)
data_wrap, labels_wrap = model.MlS()

print(data_wrap.shape, labels_wrap.shape)
```


And the output is like this:

```
cls : 0 has 65.0 samples
cls : 1 has 48.0 samples
cls : 2 has 52.0 samples
cls : 3 has 44.0 samples
cls : 4 has 48.0 samples
cls : 5 has 60.0 samples
cls : 6 has 46.0 samples
cls : 7 has 43.0 samples
self.N is 3
self.tail cls is [1, 3, 4, 6, 7]
time total knn 0.006097316741943359
(787, 10, 5) (787, 8)
```
