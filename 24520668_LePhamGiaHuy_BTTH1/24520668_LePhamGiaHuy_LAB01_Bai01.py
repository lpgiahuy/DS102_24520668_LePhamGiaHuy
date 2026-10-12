import numpy as np

# Hàm số f(x,y) = x^4 + y^4 + x^2 + y^2
def grad(u):
    x, y = u
    return np.array([
        4*x**3 + 2*x,
        4*y**3 + 2*y,
    ])

def hessian(u):
    x, y = u
    return np.array([
        [12*x**2 + 2, 0],
        [0, 12*y**2 + 2]
    ])

def newton_optim(u_0, grad_fn, hessian_fn, epsilon=1e-8, max_iter=100):
    u = u_0
    for _ in range(max_iter):
        grad = grad_fn(u)
        hessian = hessian_fn(u)

        t = np.linalg.solve(hessian, -grad)
        u = u + t

        if np.linalg.norm(t) < epsilon:
            break
    return u

if __name__ == "__main__":
    u_0 = (3, 2)
    u_optim = newton_optim(u_0, grad, hessian)
    print("Diem toi uu tim duoc:", u_optim)