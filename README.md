
# knn

### Intro
* Implemented the KNN algorithm to calculate the train the KNN model to predict the class of two datasets(crx and lenses).


### Python Files
* **preprocess.py**: Inputed the missing values and using z-norm to normalize the data.
* **knn.py**: Run knn-algorithm on the data.


### Intruction
* Preprocess the crx dataset
```
./process crx.data.training crx.data.testing
``` 

* Run KNN algorithm on crx data and lenses data
```
./run k lenses.training lenses.testing
./run k crx.training.processed crx.testing.processed
```
Replace k with the number you want to test.
