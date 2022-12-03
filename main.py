import os
import matplotlib.pyplot as plt
import numpy as np
import functools
import customtkinter as ctk
import tkinter as tk


def generate_data(n=100, c=2):
    means = 12*(np.random.rand(c, 2) - 0.5)
    means = means.repeat(n, axis=0)

    data = means + np.random.randn(n*c, 2)
    col = np.zeros((n*c, 1))
    data = np.append(data, col, axis=1)
    return data


def generate_means(k=2):
    means = 12*(np.random.rand(k, 2) - 0.5)
    label = np.array([np.array(range(k))]).T
    return np.append(means, label, axis=1)


def show_plot(data, mu, i, save=False):
    plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2], s=20)
    plt.scatter(x=mu[:, 0], y=mu[:, 1], c=mu[:, 2],
                marker="*", edgecolors="black", s=100)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])

    if save:
        if not os.path.exists('./output'):
            os.mkdir('./output')
        plt.savefig(f'./output/{i}')
        plt.close()
    else:
        plt.show()


def duplicate_mu(data, mu):
    n_data = data.shape[0]
    n_mu = mu.shape[0]

    # Duplicate the values of mu along the vertical axis
    mu_v = np.apply_along_axis(functools.partial(
        np.repeat, repeats=n_data, axis=0), axis=0, arr=mu[:, :2])

    # Split the array into multiple sub-arrays vertically
    return np.vsplit(mu_v, n_mu)


def assignment(data, mu):

    mu_duplicated = duplicate_mu(data, mu)
    x = data[:, :2]
    result = np.zeros((1, data.shape[0]))

    for i in range(mu.shape[0]):
        diff = x-mu_duplicated[i]
        result = np.vstack([result, np.apply_along_axis(
            lambda a: a[0]**2+a[1]**2, 1, diff)])
    result = result[1:]
    arg_min = np.argmin(result, 0)

    data[:, 2] = arg_min

    return data


def helper_sum(data, k):
    result = np.zeros((k, 2))
    result[int(data[2])] = [data[0], data[1]]
    return result


def get_linked_data(data, k):

    sum_data_linked = np.zeros((1, k))
    for row in data:
        sum_data_linked[0, int(row[2])] += 1

    return sum_data_linked


def centroid_update(data, k):
    linked_data = get_linked_data(data, k)
    new_mu = np.apply_along_axis(functools.partial(helper_sum, k=k), 1, data)

    new_mu = np.sum(new_mu, axis=0)
    new_mu = np.divide(new_mu, linked_data.repeat(2, 0).T, out=np.zeros_like(
        new_mu), where=linked_data.repeat(2, 0).T != 0)
    label = np.array([np.array(range(k))]).T
    new_mu = np.append(new_mu, label, axis=1)

    return new_mu


def cluster(data, mu):
    pass


def login():
    print("Logging in...")


if __name__ == "__main__":

    k = 3
    n = 100
    c = k

    data = generate_data(n, c)
    mu = generate_means(k)
    show_plot(data, mu, 0, save=True)

    for i in range(16):

        data = assignment(data, mu)
        show_plot(data, mu, i, save=True)
        mu = centroid_update(data, k)
        i += 1
        show_plot(data, mu, i, save=True)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    root.geometry("500x350")
    frame = ctk.CTkFrame(master=root)
    frame.pack(pady=20, padx=60, fill="both", expand=True)
    label = ctk.CTkLabel(master=frame, text="Login System",
                         font=("Roboto", 20))
    label.pack(pady=12, padx=10)
    entry1 = ctk.CTkEntry(master=frame, placeholder_text="Username")
    entry1.pack(pady=12, padx=10)

    entry2 = ctk.CTkEntry(master=frame, placeholder_text="Password", show="*")
    entry2.pack(pady=12, padx=10)

    button = ctk.CTkButton(master=frame, text="Login", command=login)
    button.pack(pady=12, padx=10)

    checkbox = ctk.CTkCheckBox(master=frame, text="Remember me")
    checkbox.pack(pady=12, padx=10)

    root.mainloop()
