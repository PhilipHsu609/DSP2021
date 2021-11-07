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
    """
        X: N by n matrix
        y: N by 1 vector
        target: numbers that we want to extract

        Return: (X, y) after extracting images

    """
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


def OMP(X: np.ndarray, y: np.ndarray, k: int) -> tuple:
    """
        X: n by N matrix
        y: n by 1 vector
        k: number of bases
        Return :(bases (n by k), coefficients (k by 1))
    """
    dictionary = np.ndarray.copy(X)  # dictionary nxN
    r = np.ndarray.copy(y)  # residue nx1
    l = 0
    c = None  # coefficient nx1
    bases = None  # sparse bases
    mask = np.ones(dictionary.shape[1], dtype=bool)

    # bases pre-normalize
    d_norm = np.linalg.norm(dictionary, axis=0)
    for i in range(d_norm.shape[0]):
        if d_norm[i] != 0:
            dictionary[:, i] = dictionary[:, i] / d_norm[i]
    r = r / np.linalg.norm(r)

    while l != k:
        # Find basis
        b = dictionary[:, mask].T.dot(r)
        s = np.argmax(np.abs(b))
        s = np.arange(dictionary.shape[1])[mask][s]
        mask[s] = False

        # Append basis
        if bases is None:
            bases = dictionary[:, s].reshape(-1, 1)
        else:
            bases = np.hstack((bases, dictionary[:, s].reshape(-1, 1)))

        # Calculate coefficient
        c = np.linalg.inv(bases.T.dot(bases)).dot(bases.T).dot(y)

        l += 1

    return (bases, c)


def my_Lasso(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4) -> np.ndarray:
    """
        Implement Lasso from ppt: Sparse representation L1-norm solutions p.11

        y \approx X * coefficients

        X: n by N matrix
        y: n by 1 vector
        alpha: L1-norm penalty
        max_iter: max iterations
        tol: tolerance

        Return: coefficients(N by 1)
    """
    def S(a: float, x: float) -> float:
        return np.sign(x) * max(abs(x) - a, 0)

    N = X.shape[1]
    last_c = 0
    cur_c = np.zeros((X.shape[1], 1))
    converge = False
    iters = 1

    while not converge and iters <= max_iter:
        for i in range(cur_c.shape[0]):
            r = y - X.dot(cur_c) - cur_c[i] * X[:, i].reshape(-1, 1)
            cur_c[i] = S(alpha, X[:, i].T.dot(r) / N)
            iters += 1

        if abs(last_c - cur_c).sum() < tol:
            converge = True

        last_c = cur_c.copy()

    return cur_c


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
    images, _ = extract(X, y, 5)

    eigval, eigvec = PCA(images)
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
    images, _ = extract(X, y, 5)

    eigval, eigvec = PCA(images)
    idx = np.argsort(eigval)[::-1]

    dims = [3, 10, 30, 100]

    fig, axes = plt.subplots(1, 5)
    # Plot the first "5"
    axes[0].imshow(images[0, :].reshape(28, 28), "gray")
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

    X_extract, y_extract = extract(X[:N, :], y[:N], labels)

    eigval, eigvec = PCA(X_extract)
    idx = np.argsort(eigval)[::-1]

    B = eigvec[:, idx[:dim]]
    proj = X_extract.dot(B)

    for i in range(proj.shape[0]):
        if y_extract[i] in labels:
            target[y_extract[i]].append(proj[i, :])

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
    bases = X[:N, :]
    three = X[N, :]

    sparse_bases, _ = OMP(bases.T, three.reshape(-1, 1), 5)

    fig, axes = plt.subplots(1, 5)
    for i in range(5):
        axes[i].imshow(sparse_bases[:, i].reshape(28, 28), 'gray')
        axes[i].set_title(f"basis {i+1}")
        axes[i].axis("off")

    fig.savefig("./fig/Q5.jpg", bbox_inches="tight")
    plt.close(fig)


@doc
def Q6(X: np.ndarray):
    N = 10000
    sparsity = [5, 10, 40, 200]
    bases = X[:N, :]
    eight = X[N + 1, :]

    fig, axes = plt.subplots(1, 5)
    fig.subplots_adjust(wspace=1.5)

    axes[0].imshow(eight.reshape(28, 28), "gray")
    axes[0].axis("off")

    for i, k in enumerate(sparsity):
        sparse_bases, coef = OMP(bases.T, eight.reshape(-1, 1), k)
        R = sparse_bases.dot(coef)
        axes[i + 1].imshow(R.reshape(28, 28), "gray")
        axes[i + 1].set_title(
            "L2-norm=\n{:.2f}".format(reconstruct_error(R.T, eight.reshape(1, -1))))
        axes[i + 1].axis("off")

    fig.savefig("./fig/Q6.jpg", bbox_inches="tight")
    plt.close(fig)


@doc
def Q7(X: np.ndarray, y: np.ndarray):
    X_extract, _ = extract(X, y, 8)

    # Q7 - 1
    eigval, eigvec = PCA(X_extract)
    idx = np.argsort(eigval)[::-1]
    eight = X_extract[-1, :]

    B = eigvec[:, idx[:5]]
    reconstruct_eight = eight.dot(B).dot(B.T) + X_extract.mean(axis=0)

    plt.imshow(reconstruct_eight.reshape(28, 28), "gray")
    plt.axis("off")
    plt.savefig("./fig/Q7-1.jpg", bbox_inches="tight")
    plt.close()

    # Q7 - 2
    bases = X_extract[:-1, :]
    sparse_bases, coef = OMP(bases.T, eight.reshape(-1, 1), 5)

    plt.imshow(sparse_bases.dot(coef).reshape(28, 28), "gray")
    plt.axis("off")
    plt.savefig("./fig/Q7-2.jpg", bbox_inches="tight")

    # Q7 - 3
    # from sklearn.decomposition import sparse_encode
    from sklearn.linear_model import Lasso

    lasso_params = {
        "alpha": 1,  # default = 1
        "max_iter": 1000,  # default = 1000
    }

    # Normalize
    n_bases = np.ndarray.copy(bases)
    b_norm = np.linalg.norm(bases, axis=0)
    for i in range(b_norm.shape[0]):
        if b_norm[i] != 0:
            n_bases[i, :] = n_bases[i, :] / b_norm[i]
    n_eight = eight / np.linalg.norm(eight)

    # code = sparse_encode(n_eight.reshape(1, -1), n_bases, **lasso_params)
    code = Lasso(**lasso_params).fit(n_bases.T, n_eight.reshape(-1, 1))

    # plt.imshow(code.dot(n_bases).reshape(28, 28), "gray")
    plt.imshow(n_bases.T.dot(code.coef_).reshape(28, 28), "gray")
    plt.axis("off")
    plt.savefig("./fig/Q7-3.jpg", bbox_inches="tight")
    plt.close()

    # Q7 - 4
    lasso_params["alpha"] = 0.1

    # code = sparse_encode(n_eight.reshape(1, -1), n_bases, **lasso_params)
    code = Lasso(**lasso_params).fit(n_bases.T, n_eight.reshape(-1, 1))

    # plt.imshow(code.dot(n_bases).reshape(28, 28), "gray")
    plt.imshow(n_bases.T.dot(code.coef_).reshape(28, 28), "gray")
    plt.axis("off")
    plt.savefig("./fig/Q7-4.jpg", bbox_inches="tight")
    plt.close()
