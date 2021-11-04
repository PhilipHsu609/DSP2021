import numpy as np
import matplotlib.pyplot as plt
from typing import Union


def doc(func):
    def n_func(*args):
        print("\nRunning ", func.__name__, "...")
        func(*args)
        print(func.__name__, " complete...\n")
    return n_func


def extract(X: np.ndarray, y: np.ndarray, target: Union[int, list]) -> tuple:
    """Return: (X, y) after extracting images"""
    X2 = []
    y2 = []

    if isinstance(target, int):
        target = [target]

    for i in range(X.shape[0]):
        if y[i] in target:
            X2.append(X[i, :])
            y2.append(y[i])

    return (np.array(X2), np.array(y2))


def PCA(X: np.ndarray) -> tuple:
    """Return: (eigenvalues, eigenvectors)"""
    # Centered
    S = X - X.mean(axis=0)

    # Scattered matrix (Covariance matrix)
    S = np.cov(S, rowvar=False)

    # Find eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eigh(S)

    return (eigval, eigvec)


def OMP(X: np.ndarray, target: np.ndarray, k: int) -> tuple:
    """Return :(bases, coefficients)"""
    B = np.ndarray.copy(X)  # dictionary
    r = np.ndarray.copy(target)  # residue
    l = 0
    c = None  # coefficient
    bases = None  # sparse bases
    mask = np.ones(B.shape[0], dtype=bool)

    # Normalize
    norm = np.linalg.norm(B, axis=1)
    for i in range(norm.shape[0]):
        if norm[i] != 0:
            B[i, :] = B[i, :] / norm[i]
    r = r / np.linalg.norm(r)

    while l != k:
        # Find basis
        b = B[mask, :].dot(r.T)
        s = np.argmax(np.abs(b))
        s = np.arange(B.shape[0])[mask][s]
        mask[s] = False

        # Append basis
        if bases is None:
            bases = B[s, :].reshape((1, B.shape[1]))
        else:
            bases = np.vstack((bases, B[s, :]))

        # Calculate coefficient
        c = np.linalg.inv(bases.dot(bases.T)).dot(bases).dot(target.T)

        l += 1

    return (bases, c)


def reconstruct_error(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.square(x - y).sum())


@doc
def Q1(X: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(X.mean(axis=0).reshape(28, 28), "gray")
    ax.axis("off")
    fig.savefig("./fig/Q1.jpg", bbox_inches="tight")
    plt.close(fig)


@doc
def Q2(X: np.ndarray, y: np.ndarray):
    M, _ = extract(X, y, 5)

    eigval, eigvec = PCA(M)
    idx = np.argsort(eigval)[::-1]

    fig, axes = plt.subplots(1, 3)

    for i in range(3):
        axes[i].imshow(eigvec[:, idx[i]].reshape(28, 28), "gray")
        axes[i].set_title("$\\lambda$=\n{:.2f}".format(eigval[idx[i]]))
        axes[i].axis("off")

    fig.savefig("./fig/Q2.jpg", bbox_inches="tight")
    plt.close(fig)


@doc
def Q3(X: np.ndarray, y: np.ndarray):
    M, _ = extract(X, y, 5)

    eigval, eigvec = PCA(M)
    idx = np.argsort(eigval)[::-1]

    dims = [3, 10, 30, 100]

    fig, axes = plt.subplots(1, 5)
    # Plot the first "5"
    axes[0].imshow(M[0, :].reshape(28, 28), "gray")
    axes[0].axis("off")

    for i, dim in enumerate(dims):
        B = eigvec[:, idx[:dim]]
        R = X.dot(B).dot(B.T) + X.mean(axis=0)
        axes[i + 1].imshow(R[0, :].reshape(28, 28), "gray")
        axes[i + 1].axis("off")

    fig.savefig("./fig/Q3.jpg", bbox_inches="tight")
    plt.close(fig)


@doc
def Q4(X: np.ndarray, y: np.ndarray):
    labels = [1, 3, 6]
    target = {label: [] for label in labels}
    color = ["red", "green", "blue"]
    dim = 2
    N = 10000

    X2, y2 = extract(X[:N, :], y[:N], labels)

    eigval, eigvec = PCA(X2)
    idx = np.argsort(eigval)[::-1]

    B = eigvec[:, idx[:dim]]
    proj = X2.dot(B)

    for i in range(proj.shape[0]):
        if y2[i] in labels:
            target[y2[i]].append(proj[i, :])

    fig, ax = plt.subplots()
    for i, label in enumerate(labels):
        target[label] = np.array(target[label])
        ax.scatter(
            target[label][:, 0],
            target[label][:, 1],
            c=color[i],
            label=label
        )

    ax.legend()
    fig.savefig("./fig/Q4.jpg", bbox_inches="tight")
    plt.close(fig)


@doc
def Q5(X: np.ndarray):
    N = 10000
    B = X[:N, :]
    three = X[N, :]

    bases, _ = OMP(B, three, 5)

    fig, axes = plt.subplots(1, 5)
    for i in range(5):
        axes[i].imshow(bases[i, :].reshape(28, 28), 'gray')
        axes[i].set_title(f"basis {i+1}")
        axes[i].axis("off")

    fig.savefig("./fig/Q5.jpg", bbox_inches="tight")
    plt.close(fig)


@doc
def Q6(X: np.ndarray):
    sparsity = [5, 10, 40, 200]
    bases = X[:10000, :]
    eight = X[10001, :]

    fig, axes = plt.subplots(1, 5)
    fig.subplots_adjust(wspace=1.5)

    axes[0].imshow(eight.reshape(28, 28), "gray")
    axes[0].axis("off")

    for i, k in enumerate(sparsity):
        sparse_bases, coef = OMP(bases, eight, k)
        R = sparse_bases.T.dot(coef)
        axes[i + 1].imshow(R.reshape(28, 28), "gray")
        axes[i + 1].set_title(
            "L2-norm=\n{:.2f}".format(reconstruct_error(R, eight)))
        axes[i + 1].axis("off")

    fig.savefig("./fig/Q6.jpg", bbox_inches="tight")
    plt.close(fig)


@doc
def Q7(X: np.ndarray, y: np.ndarray):
    X2, _ = extract(X, y, 8)

    # Q7 - 1
    eigval, eigvec = PCA(X2)
    idx = np.argsort(eigval)[::-1]
    last_eight = X2[-1, :]

    B = eigvec[:, idx[:5]]
    reconstruct_eight = last_eight.dot(B).dot(B.T) + X2.mean(axis=0)

    plt.imshow(reconstruct_eight.reshape(28, 28), "gray")
    plt.axis("off")
    plt.savefig("./fig/Q7-1.jpg", bbox_inches="tight")
    plt.close()

    # Q7 - 2
    bases = X2[:-1, :]
    sparse_bases, coef = OMP(bases, last_eight, 5)

    plt.imshow(sparse_bases.T.dot(coef).reshape(28, 28), "gray")
    plt.axis("off")
    plt.savefig("./fig/Q7-2.jpg", bbox_inches="tight")

    # Q7 - 3
    from sklearn.decomposition import sparse_encode

    n_bases = np.ndarray.copy(bases)

    lasso_params = {
        "algorithm": "lasso_cd",
        "alpha": 0.9,  # default = 1
        "max_iter": 1000,  # default = 1000
        "n_jobs": 8
    }

    # Normalize
    norm = np.linalg.norm(bases, axis=1)
    for i in range(norm.shape[0]):
        if norm[i] != 0:
            n_bases[i, :] = n_bases[i, :] / norm[i]
    n_eight = last_eight / np.linalg.norm(last_eight)

    code = sparse_encode(n_eight.reshape(1, -1), n_bases, **lasso_params)

    plt.imshow(code.dot(bases).reshape(28, 28), "gray")
    plt.axis("off")
    plt.savefig("./fig/Q7-3.jpg", bbox_inches="tight")
    plt.close()

    # Q7 - 4
    lasso_params["alpha"] = 0.1

    code = sparse_encode(n_eight.reshape(1, -1), n_bases, **lasso_params)

    plt.imshow(code.dot(n_bases).reshape(28, 28), "gray")
    plt.axis("off")
    plt.savefig("./fig/Q7-4.jpg", bbox_inches="tight")
    plt.close()
