# HYPERPARAMETER OPTIMIZATION USING RAYTUNE AND RAPIDS

![Header](https://miro.medium.com/max/2800/1*MgRODF1avuHXkYZQKJ_yeQ.png)

Hyperparameter optimization is a method used to enhance the accuracy of a model. Hyperparameter tuning can make the difference between an average model and a highly accurate one. The goal of this model was to predict the value of a football player by using random forest on gpu and optimized the accuray of the prediciton based on features such as rating, skill rate, work rate, attacking rate, position and etc.

Link to view the [Data](https://www.kaggle.com/karangadiya/fifa19)

**Rapids is a suite of open source software libraries and APIs gives you the ability to execute end-to-end data science and analytics pipelines entirely on GPUs. Imagine scikit-learn on steroids, that is rapids.**

Without HPO Random Forest -> [No-HPO](https://github.com/fadilparves/RAPIDS_RANDOM_FOREST_HPO/blob/master/random_forest_ori.py) (Training on GPU)

### Without HPO Random Forest
Running default setting on Random Forest
> n_estimator = 100,
> max_depth = 16,
> max_bins = 8

Accuracy -> R2 score = 75%

### Goal is to tune the model and enhance the accuracy without doing any feature engineering
Hence comes the ray tune module into the picture. [Ray.Tune](https://docs.ray.io/en/latest/tune/index.html)

[HPO-code](https://github.com/fadilparves/RAPIDS_RANDOM_FOREST_HPO/blob/master/random_forest_hpo.py)

#### Ray Tune Config
> number of samples = 10,
> number of folds = 3,
> range for n_estimators = 500 - 1500,
> range for max_depth = 10 - 20,
> range for max_features = 0.5 - 1.0,
> n_bins = 18

**_Configuration that is done here are restricted as the machine used for this experiment is only running on GTX1060. If you have more powerful GPU, then you may have wider range of configuration to test._**

Ray will randomly select any value from the range and add into the model as hyperparameter
```python
self.rf_model = curfc(
                n_estimators=self._model_params["n_estimators"],
                max_depth=self._model_params["max_depth"],
                n_bins=self._model_params["n_bins"],
                max_features=self._model_params["max_features"],
            )
```
Total of 300 samples executed, but some iterations stopped early due to early stopping conditions.
```
+---------------------+------------+-------+-------------+----------------+----------------+--------+------------------+
| Trial name          | status     | loc   |   max_depth |   max_features |   n_estimators |   iter |   total time (s) |
|---------------------+------------+-------+-------------+----------------+----------------+--------+------------------|
| WrappedTrainable_1  | TERMINATED |       |     13.7454 |       0.975357 |       1231.99  |      3 |         234.135  |
| WrappedTrainable_2  | TERMINATED |       |     15.9866 |       0.578009 |        655.995 |      1 |          35.5271 |
| WrappedTrainable_3  | TERMINATED |       |     10.5808 |       0.933088 |       1101.12  |      1 |          58.8539 |
| WrappedTrainable_4  | TERMINATED |       |     17.0807 |       0.510292 |       1469.91  |      1 |          98.2842 |
| WrappedTrainable_5  | TERMINATED |       |     18.3244 |       0.60617  |        681.825 |      3 |         180.687  |
| WrappedTrainable_6  | TERMINATED |       |     11.834  |       0.652121 |       1024.76  |      3 |         124.095  |
| WrappedTrainable_7  | TERMINATED |       |     14.3195 |       0.645615 |       1111.85  |      3 |         149.505  |
| WrappedTrainable_8  | TERMINATED |       |     11.3949 |       0.646072 |        866.362 |      1 |          36.0093 |
| WrappedTrainable_9  | TERMINATED |       |     14.5607 |       0.892588 |        699.674 |      1 |          43.3045 |
| WrappedTrainable_10 | TERMINATED |       |     15.1423 |       0.796207 |        546.45  |      3 |         112.048  |
+---------------------+------------+-------+-------------+----------------+----------------+--------+------------------+
```
Output for all parameters are stored in [trials.csv](https://github.com/fadilparves/RAPIDS_RANDOM_FOREST_HPO/blob/master/trials.csv)

The best performing parameters were experiment number 6:
> max_depth=11, max_features=0.6, n_estimators=1024

With these hyperparameters the model accuracy increased to 83%. As you can see by finding better parameters we can make the model more accurate, now all is left is to work on feature engineering and re run the HPO to increase the accuracy more.

## Those interested to try and run this
1. Install [rapids](https://rapids.ai/start.html)
2. Install ray ```pip install 'ray[tune]' torch torchvision```
3. Clone this repo
4. Run ```python/python3 random_forest_hpo.py```

### References
1. [rapids example](https://github.com/rapidsai/cloud-ml-examples/blob/main/ray/notebooks/Ray_RAPIDS_HPO.ipynb)
2. [medium](https://medium.com/rapids-ai/30x-faster-hyperparameter-search-with-raytune-and-rapids-403013fbefc5)
3. [ray](https://docs.ray.io/en/latest/tune/index.html#quick-start)
4. [cuml](https://docs.rapids.ai/api/cuml/stable/api.html#random-forest)

## Contributor
<a href="https://github.com/fadilparves/RAPIDS_RANDOM_FOREST_HPO/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=fadilparves/RAPIDS_RANDOM_FOREST_HPO" />
</a>
