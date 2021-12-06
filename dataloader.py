import torch
import numpy as np
import pandas as pd
import os.path as osp

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset


class MovieLens1MAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, user_path, movie_path, sep='::'):
        rating_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        user_columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode']
        movie_columns = ['MovieID', 'Title', 'Genres']
        # data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        rating = pd.read_csv(dataset_path, sep=sep, engine='python', header=0, names=rating_columns)

        user = pd.read_csv(user_path, sep=sep, engine='python', header=0, names=user_columns)
        user['Gender'] = user['Gender'].map({'M': 0, 'F': 1})
        user['Age'] = pd.cut(
            user['Age'], bins=[user['Age'].min(), 18, 24, 34, 44, 49, 55, user['Age'].max()],
            labels=[i for i in range(7)])  # rightmost edge is included
        user['Zipcode'] = user['Zipcode'].astype('category').cat.codes
        # user = user.drop(columns=['Zipcode'])

        movie = pd.read_csv(movie_path, sep=sep, engine='python', header=0, names=movie_columns)
        movie['Year'] = movie['Title'].str[-5:-1].astype(np.int)
        movie['Year'] = pd.cut(
            movie['Year'],
            bins=[user['Age'].min(), 1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2000],
            labels=[i for i in range(9)])
        genres = movie['Genres'].str.split('|')
        unique_genres = set(genres.sum())

        for g in unique_genres:
            movie[g] = movie['Genres'].str.contains(g).astype(np.int)

        rating_final = pd.merge(rating, user)
        rating_final = pd.merge(rating_final, movie)
        rating_final = rating_final.drop(columns=['UserID', 'MovieID', 'Timestamp', 'Title', 'Genres'])
        rating = rating_final.to_numpy()
        print(rating_final.describe())

        self.items = rating[:, 0:2].astype(np.int)
        self.targets = self.__preprocess_target(rating[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1, ), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target


def get_data(dataset):
    assert dataset in [
        'criteo', 'avazu',
        'movielens-1m', 'movielens-20m', 'movielens-1m-augmented']

    path = osp.expanduser('data')

    if dataset == 'criteo':
        dataset = CriteoDataset(f'{path}/criteo/train.txt', f'{path}/criteo/.criteo')
    elif dataset == 'avazu':
        dataset = AvazuDataset(f'{path}/avazu/train', f'{path}/avazu/.avazu')
    elif dataset == 'movielens-1m':
        dataset = MovieLens1MDataset(f'{path}/movielens/ml-1m/ratings.dat')
    elif dataset == 'movielens-20m':
        dataset = MovieLens20MDataset(f'{path}/movielens/ml-20m/ratings.csv')
    elif dataset == 'movielens-1m-augmented':
        dataset = MovieLens1MAugmentedDataset(
            f'{path}/movielens/ml-1m/ratings.dat',
            f'{path}/movielens/ml-1m/users.dat',
            f'{path}/movielens/ml-1m/movies.dat'
        )
    return dataset
