

from data_loader import ready_movielens_merge_data
from gae import train


if __name__ == '__main__':
    data = ready_movielens_merge_data()
    train(data)
    