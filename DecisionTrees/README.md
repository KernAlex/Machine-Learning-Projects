# Decision Tree / Random Forest

This simple example is currently set up to split on two classes. More will be added when I have the time.

## Usage

```python
''' 
X: Dataset matrix to train on, must be vectors of reals
y: labels in the form of 0, 1
names: list of feature names, recommend to be strings (optional)
'''
dTree = DecisionTree(X, features=names)
dTree.fit(X, y, 5)
train = [dTree.predict(X[i]) for i in range(len(X))]
 # Prints the tree, if feature names were provided it will show the names of the splits
print(dTree)
length = len(train) # prints training accuracy 
count = 0
for i in range(length):
      if y[i] == train[i]:
           count += 1
print(count/length)
```

## Contributing
Some of the skeleton code was provided, but the design is completely mine.
