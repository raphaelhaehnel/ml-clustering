import os
import matplotlib.pyplot as plt
import numpy as np
import functools
import scipy.io
import shutil
from tqdm import tqdm

LIMIT_ITER = 100


def generate_data(n, c, dim):
    means = 12*(np.random.rand(c, dim) - 0.5)
    means = means.repeat(n, axis=0)

    data = means + np.random.randn(n*c, dim)
    col = np.zeros((n*c, 1))
    data = np.append(data, col, axis=1)
    return data


def generate_means(k, dim):
    means = 12*(np.random.rand(k, dim) - 0.5)
    label = np.array([np.array(range(k))]).T
    return np.append(means, label, axis=1)


def generate_means_pixels(k, dim):
    means = np.random.rand(k, dim)
    label = np.array([np.array(range(k))]).T
    return np.append(means, label, axis=1)


def show_plot(data, mu, i, save, dim):
    if dim == 2:
        plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2], s=20)
        plt.scatter(x=mu[:, 0], y=mu[:, 1], c=mu[:, 2],
                    marker="*", edgecolors="black", s=100)
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 3], s=20)
        ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], c=mu[:, 3],
                   marker="*", edgecolors="black", s=100)

    if save:
        if not os.path.exists('./output'):
            os.mkdir('./output')
        plt.savefig(f'./output/{i}')
        plt.close()
    else:
        plt.show()


def duplicate_mu(data, mu, dim):
    n_data = data.shape[0]
    n_mu = mu.shape[0]

    # Duplicate the values of mu along the vertical axis
    mu_v = np.apply_along_axis(functools.partial(
        np.repeat, repeats=n_data, axis=0), axis=0, arr=mu[:, :dim])

    # Split the array into multiple sub-arrays vertically
    return np.vsplit(mu_v, n_mu)


def assignment(data, mu, dim):

    mu_duplicated = duplicate_mu(data, mu, dim)
    x = data[:, :dim]
    result = np.zeros((1, data.shape[0]))

    for i in tqdm(range(mu.shape[0])):
        diff = x-mu_duplicated[i]
        result = np.vstack([result, np.apply_along_axis(
            lambda a: np.sum(a**2), 1, diff)])
    result = result[1:]
    arg_min = np.argmin(result, 0)

    data[:, dim] = arg_min
    cost = np.sum(np.min(result, 0))

    return data, cost


def helper_sum(data, k, dim):
    result = np.zeros((k, dim))
    result[int(data[dim])] = data[:-1]

    return result


def cout_data_labels(data, k, dim):

    count = np.zeros((1, k))
    for row in data:
        count[0, int(row[dim])] += 1

    return count


def centroid_update(data, k, dim):
    cout_labels = cout_data_labels(data, k, dim)
    new_mu = np.apply_along_axis(
        functools.partial(helper_sum, k=k, dim=dim), 1, data)

    new_mu = np.sum(new_mu, axis=0)
    new_mu = np.divide(new_mu, cout_labels.repeat(dim, 0).T, out=np.zeros_like(
        new_mu), where=cout_labels.repeat(dim, 0).T != 0)
    label = np.array([np.array(range(k))]).T
    new_mu = np.append(new_mu, label, axis=1)

    return new_mu


def run_clustering(data, k, mu, dim, save):

    total_cost = np.array([])

    for i in range(LIMIT_ITER):
        data, cost = assignment(data, mu, dim)
        total_cost = np.append(total_cost, cost)
        new_mu = centroid_update(data, k, dim)
        if np.array_equal(new_mu, mu):
            break
        mu = new_mu
        show_plot(data, mu, i, save=save, dim=dim)

    return data, total_cost


def success_rate(y_train, y_output):
    return np.count_nonzero(y_output == y_train)/len(y_output)*100


if __name__ == "__main__":

    k = 5
    c = k

    if os.path.exists('./output'):
        shutil.rmtree('./output')

    n = 100
    dim = 3
    data = generate_data(n, c, dim)
    mu = generate_means(k, dim)

    output, total_cost = run_clustering(data, k, mu, dim, save=True)
    # result_success = success_rate(y_train, output[:, -1])
    # print(f'Success rate = {result_success}%')
    plt.plot(total_cost, marker="o")
    plt.show()
