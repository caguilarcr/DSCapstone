# DSCapstone

This project builds a movie recommendation on the [10M MovieLens Dataset](https://grouplens.org/datasets/movielens/10m/) using five different approaches. The dataset contains data collected by the Movie Lens project and has around 10 million movie ratings given by anonymized users, the movie's genres, title, and release year.

The objective of the project is to produce a system that predicts the rating that a user will give to a particular movie with an RMSE (Root Mean Square Error) below to 0.86490. To achieve this, we will analyze five methods that incrementally consider more information.

The first method would be to predict the simple mean for every movie. The second will add a variable to handle movie bias. The third will add a control for user bias. The last two methods will consider movie and user bias, respectively, but regularizing the results to penalize higher residuals from few ratings.

We are going to use the following packages to perform the analysis

1. [Tidyverse](https://www.tidyverse.org/)
2. [Caret](http://topepo.github.io/caret/index.html)
3. [Data.table](https://cran.r-project.org/web/packages/data.table/index.html)