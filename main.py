import numpy as np
import math
import plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def cholesky_method(A, b):
    row_count = A.shape[0]
    column_count = A.shape[1]
    i = j = 0
    L = np.zeros((row_count, column_count))
    while j < column_count:
        i = 0
        while i < row_count:
            if i == j == 0:
                L[i][i] = math.sqrt(A[i][j])
            if i == 0 and j > 0:
                L[j][i] = A[j][i] / L[0][0]

            if i > 0 and i < row_count  - 1:
                temp = 0
                for k in range(0, i):
                    temp = temp + math.pow(L[i][k], 2)
                L[i][i] = math.sqrt(A[i][i] - temp)
            elif j > 0 and i > 0:
                temp = 0
                for k in range(0, i):
                    temp = temp + (L[j][k] + L[i][k])

                L[j][i] = (1 / L[i][i]) * (A[j][i] - temp)
            if j == column_count and i == row_count:
                temp = 0
                for k in range(0, row_count - 1):
                    math.pow(L[row_count - 1][k], 2)
                L[row_count - 1][row_count - 1] = math.sqrt(A[row_count - 1][column_count - 1] - temp)
            i = i + 1
        j = j + 1


def crouts_method(A, b):
    row_count = A.shape[0]
    column_count = A.shape[1]
    i = j = 0

    L = np.zeros((row_count, column_count))
    U = np.identity(row_count, dtype=float)

    while i < row_count:
        j = 0
        while j < column_count:
            if j == 0:
                L[i][j] = A[i][0]
            if i == 0:
                U[i][j] = (A[i][j] / A[0][0])
            if i >= j:
                temp = 0
                for k in range(0, j):
                    temp = temp + (L[i][k] * U[k][j])

                L[i][j] = A[i][j] - temp
            if i < j:
                temp = 0
                for k in range(0, i ):
                    temp = temp + (L[i][k] * U[k][j])

                U[i][j] = (1 / L[i][i]) * (A[i][j] - temp)
            j = j + 1
        i = i + 1


    # Forward substitution
    L = np.c_[L, b]
    i = 0
    Y = np.array([])
    while i < L.shape[0]:
        diagonal_element = L[i][i]
        if i > 0:
            L[i][: i] = L[i][: i] * Y

        for j in range(0, i):
            L[i][L.shape[1] - 1] = L[i][L.shape[1] - 1] - L[i][j]

        L[i][L.shape[1] - 1] = L[i][L.shape[1] - 1] / diagonal_element
        Y = np.append(Y, L[i][L.shape[1] - 1])
        i = i + 1

    # Backward substitution
    i = A.shape[0] - 1
    X = np.array([])

    U = np.c_[U, Y]
    while i >= 0:
        diagonal_element = U[i][i]
        if i < A.shape[0] - 1:
            U[i][i + 1:U.shape[0]] = U[i][i + 1:U.shape[0]] * X

        for j in range(i + 1, U.shape[0]):
            U[i][U.shape[1] - 1] = U[i][U.shape[1] - 1] - U[i][j]

        U[i][U.shape[1] - 1] = U[i][U.shape[1] - 1] / diagonal_element
        X = np.insert(X, 0, U[i][U.shape[1] - 1], axis=0)
        i = i - 1
    print("Hello")


def do_little(A, b):

    L = np.identity(A.shape[0], dtype=float)
    # Forward elimination
    i = 0
    while i < A.shape[0]:
        diagonal_element = A[i][i]
        print("Diagonal element: {}".format(diagonal_element))
        # GIEVEN THAT THE DIAGONAL ELEMENT IS ZERO
        # SET THE LOWER ELEMENTS TO ZERO
        if diagonal_element != 0:
            for j in range(i + 1, A.shape[0]):

                m = A[j][i] / diagonal_element
                L[j][i] = m
                print("multiplier: m[{}][{}] = {}".format(i, j, m))
                reduced_row = A[j] - (m * A[i])
                A[j] = reduced_row
            i = i + 1
        else:
            for k in range(i + 1, A.shape[0]):
                if A[k][i] != 0:
                    temp = np.array(A[i])
                    A[i] = np.array(A[k])
                    A[k] = np.array(temp)
                    break
    #Forward substitution
    L = np.c_[L, b]
    i = 0
    Y = np.array([])
    while i < L.shape[0]:
        diagonal_element = L[i][i]
        if i > 0:
            L[i][: i] = L[i][: i] * Y


        for j in range(0, i):
            L[i][L.shape[1] - 1] = L[i][L.shape[1] - 1] - L[i][j]

        L[i][L.shape[1] - 1] = L[i][L.shape[1] - 1] / diagonal_element
        Y = np.append(Y,  L[i][L.shape[1] - 1])
        i = i + 1

    #Backward substitution
    i = A.shape[0] - 1
    X = np.array([])

    A = np.c_[A, Y]
    while i >= 0:
        diagonal_element = A[i][i]
        if i < A.shape[0] - 1:
            A[i][i + 1:A.shape[0]] = A[i][i + 1:A.shape[0]] * X

        for j in range(i + 1, A.shape[0]):
            A[i][A.shape[1] - 1] = A[i][A.shape[1] - 1] - A[i][j]

        A[i][A.shape[1] - 1] = A[i][A.shape[1] - 1] / diagonal_element
        X = np.insert(X, 0, A[i][A.shape[1] - 1], axis=0)
        i = i - 1
    print(A)
    print(L)


def gauss_jordan():

    A = np.array([
        [1., 2., 0., 3],
        [-1., 0., -2., -5.],
        [-3., -5., 1., -4.]
    ])

    i = 0
    while i < A.shape[0]:

        A[i] = A[i] / A[i][i]
        diagonal_element = A[i][i]
        for j in range(i + 1, A.shape[0]):
            m = A[j][i] / diagonal_element
            reduced_row = A[j] - (m * A[i])
            A[j] = reduced_row
        i = i + 1

    i = A.shape[0] - 1
    while i >= 0:
        current_row = A[i]
        print("current row: {}".format(current_row))
        j = i - 1
        while j >= 0:
            A[j] = A[j] - (current_row * A[j][i])
            j = j- 1
        i = i - 1


def test():
    N = 3
    A = np.eye(N)
    A = np.c_[A, np.ones(N)]
    print("a: {}".format(A))


def forward_elimination(A):
    '''
    Given a coefficient matrix the following python function will reduce the matrix into an u
    upper triangular form.

    Parameters
    ----------
    A: numpy matrix
        The original matrix to be reduced to a lower triangular form.
    '''

    #initialize counter to zero and get the row count of the matrix
    i = 0
    n = A.shape[0]
    while i < n:
        diagonal_element = A[i][i]
        # GIEVEN THAT THE DIAGONAL ELEMENT IS ZERO
        # SET THE LOWER ELEMENTS TO ZERO
        if diagonal_element != 0:

            for j in range(i + 1, n):
                m = A[j][i] / diagonal_element
                reduced_row = A[j] - (m * A[i])
                A[j] = reduced_row
            i = i + 1
        else:
            #Swap rows if a diagonal element with zero entry detected
            for k in range(i + 1, n):
                if A[k][i] != 0:
                    temp = np.array(A[i])
                    A[i] = np.array(A[k])
                    A[k] = np.array(temp)
                    break

    return A


def backward_substitution(A):
    '''
    Given a matrix the function will apply the backward substitution to find the solutions
    for the given set of reduced equations.
    :param A: numpy array
        The upper triangular matrix.
    :return: numpy array
        The solution for the set of equations.
    '''


    i = A.shape[0] - 1
    X = np.array([])
    while i >= 0:
        diagonal_element = A[i][i]
        if i < A.shape[0] - 1:
            A[i][i + 1:A.shape[0]] = A[i][i + 1:A.shape[0]] * X

        for j in range(i + 1, A.shape[0]):
            A[i][A.shape[1] - 1] = A[i][A.shape[1] - 1] - A[i][j]

        A[i][A.shape[1] - 1] = A[i][A.shape[1] - 1] / diagonal_element
        X = np.insert(X, 0, A[i][A.shape[1] - 1], axis=0)
        i = i - 1

    return X


def simple_gaussian_elimination_method(A):


    reduced_matrix = forward_elimination(A)

    solution = backward_substitution(reduced_matrix)

    for i in range(0, solution.shape[0]):
        print("X{} = {}".format(i + 1, solution[i]))

    return solution


def jacobi_method(A, b, x, max_iter, acc):
    n = x[0].shape[0]
    error = np.array([])
    for k in range(0, max_iter):

        x_temp = np.array([])
        for i in range(0, n):

            sum1 = 0
            for j in range(0, i):
                sum1 = sum1 + (A[i][j] * x[k][j])

            sum2 = 0
            for j in range(i + 1, n):
                sum2 = sum2 + (A[i][j] * x[k][j])

            temp = (1 / A[i][i]) * (b[i] - sum1 - sum2)

            x_temp = np.append(x_temp, temp)

        current_erorr = np.amax(x_temp - x[x.shape[0] - 1])
        error = np.append(error, current_erorr)
        if current_erorr <= acc:
            break
        x = np.vstack([x, x_temp])
        return x


def gauss_seidel(A, b, x, max_iter, acc):
    n = x[0].shape[0]
    error = np.array([])
    for k in range(0, max_iter):
        x = np.vstack([x, np.array([np.zeros(n)])])
        for i in range(0, n):
            sum1 = 0
            for j in range(0, i):
                sum1 = sum1 + (A[i][j] * x[k + 1][j])
            sum2 = 0
            for j in range(i + 1, n):
                sum2 = sum2 + ( A[i][j] * x[k][j])

            temp = (1 / A[i][i]) * (b[i] - sum1 - sum2)

            x[k + 1][i] = temp

        current_erorr = np.amax(x[k + 1] - x[k])
        error = np.append(error, current_erorr)

        if current_erorr <= acc:
            break

    return error

def successive_over_relaxation(A, b, x, w, max_iter, acc):
    n = x[0].shape[0]
    error = np.array([])
    for k in range(0, max_iter):
        x = np.vstack([x, np.array([np.zeros(n)])])
        for i in range(0, n):
            sum1 = 0
            for j in range(0, i):
                sum1 = sum1 + (A[i][j] * x[k + 1][j])
            sum2 = 0
            for j in range(i + 1, n):
                sum2 = sum2 + (A[i][j] * x[k][j])

            temp = (1 - w) * x[k][i] + (w / A[i][i]) * (b[i] - sum1 - sum2)

            x[k + 1][i] = temp

        current_erorr = np.amax(x[k + 1] - x[k])
        error = np.append(error, current_erorr)
        if current_erorr <= acc:
            break

    return error


def show_plot(title, x_axis_label, y_axis_label, data,  g_type='spline'):
    layout = go.Layout(
        title=title,
        yaxis=dict(
            title=x_axis_label
        ),
        xaxis=dict(
            title=y_axis_label
        )
    )

    traces = []

    for i in data:
        traces.append(
            go.Scatter(
            x = i[1],
            y= i[2],
            mode= "lines",
            name= i[0],
            line=dict(
                shape=g_type
            )
        ))


    fig = go.Figure(data=traces, layout=layout)
    return py.offline.plot(fig)


def data_plot():
    data = { "2019": [
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,11,9,0,24,52,50,29,21,91,1,34,44,37],
            [76,55,4,36,54,53,30,11,58,0,12,102,98,115,101,138,84,73,95,82,81,80,92,69,103,120,72,90,0,0],
            [91,101,1,20,85,90,103,78,55,61,57,41,74,74,85,94,0,55,52,47,77,83,57,77,0,0,0,0,0,0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
    }
    N = 1000
    x = np.linspace(1, 30, 30)

    january_call_count = np.array(data["2019"][0])
    february_call_count = np.array(data["2019"][1])
    march_call_count = np.array(data["2019"][2])


    layout = go.Layout(
        title = "Monthly Scatter Plot 2019",
        yaxis = dict(
            title = "Number of calls"
        ),
        xaxis = dict(
            title = "Number of days in month"
        )
    )

    trace_january = go.Scatter(
        name="January",
        x=x,
        y=january_call_count,
        mode='lines+markers'
    )

    trace_february = go.Scatter(
        name="February",
        x=x,
        y=february_call_count,
        mode='lines+markers'
    )

    trace_march = go.Scatter(
        name="March",
        x=x,
        y=march_call_count,
        mode='lines+markers'
    )

    data = [trace_january, trace_march, trace_february]

    fig = go.Figure(data = data, layout=layout)


    # Plot and embed in ipython notebook!
    py.offline.plot(fig, filename='basic-scatter')


def curve_fitting_linear():
    y = np.array([76, 53, 102, 73, 69, 0])
    x = np.linspace(1, y.size, y.size) #GENERATE THE INDEPENDENT VARIABLE FROM THE DEPENDENT DATA POINT


    n = x.size
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_sqr = np.sum(x * x)

    #NOW WE CAN APPLY GAUSS JORDAN TO SOLVE FOR A AND B FOR THE EQUATION Y = ax + b

    print("x: {}, y: {}, n: {}, sum_x: {}, sum_y: {}, sum_x_sqr: {}, sum_xy: {}".format(x, y, n, sum_x, sum_y, sum_x_sqr, sum_xy))


    # y = mx + b
    #
    A = np.array([
        [sum_x, n, sum_y],
        [sum_x_sqr, sum_x, sum_xy]
    ], dtype=float)

    solution = simple_gaussian_elimination_method(A)

    def f(x):
        return (solution[0] * x) + solution[1]

    vfunc = np.vectorize(f)

    x_ = np.linspace(1, 6, 100)
    result = vfunc(np.array(x_))

    plt.plot(x.tolist(), y.tolist(), label="real data")
    plt.plot(x_.tolist(), result.tolist(), label="interpolated data")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Non linear curve fitting')
    plt.legend()
    plt.show()

    print("solution: {}".format(solution))

def curve_fitting_non_linear():
    y = np.array([76, 53, 102, 73, 69, 0])
    x = np.linspace(1, y.size, y.size)  # GENERATE THE INDEPENDENT VARIABLE FROM THE DEPENDENT DATA POINT

    n = x.size
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)

    sum_x_sqr = np.sum(x * x)
    sum_x_cube = np.sum(x * x * x)
    sum_x_pw4 = np.sum(x * x * x * x)
    sum_x_sqr_y = np.sum((x * x) * y)


    A = np.array([
        [sum_x_sqr, sum_x, n, sum_y],
        [sum_x_cube, sum_x_sqr, sum_x, sum_xy],
        [sum_x_pw4, sum_x_cube, sum_x_sqr, sum_x_sqr_y]
    ], dtype=float)


    solution = simple_gaussian_elimination_method(A)

    def f(x):
        return (solution[0] * (x * x)) + (solution[1] * x) + solution[2]

    vfunc = np.vectorize(f)

    x_ = np.linspace(1, 6, 100)
    result = vfunc(np.array(x_))


    plt.plot(x.tolist(), y.tolist(), label="real data")
    plt.plot(x_.tolist(), result.tolist(), label="interpolated data")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Non linear curve fitting')
    plt.legend()
    plt.show()


    print("solution: {}", solution)

# simple_gaussian_elimination_method()
# A = np.array([
#     [5, -1, 1],
#     [2, 8, -1],
#     [-1, 1, 4]
# ], dtype=float)


# A = np.array([
#     [2, 1, 0, 0],
#     [1, 2, 1, 0],
#     [0, 1, 2, 1],
#     [0, 0, 1, 2]
# ], dtype=float)

# A = np.array([
#     [9, 3, 4, 7],
#     [4, 3, 4, 8],
#     [1, 1, 1, 3]
# ], dtype=float)

#
# A = np.array([
#     [5, -1, 1],
#     [2, 8, -1],
#     [-1, 1, 4]
# ])
# b = np.array([10, 11, 3])
#
#
# b = np.array([4, 8, 12, 11], dtype=float)
#
#
# x = np.array([
#     [0, 0, 0]
# ])
#
# curve_fitting_linear()

#
# successive_over_relaxation(A, b, x, 1.25, 1000, 0)
# #simple_gaussian_elimination_method()
# #data_plot()
#
# error_jacobi = jacobi_method(A, b, x, 1000, 0)
# error_gauss_seidel = gauss_seidel(A,b, x, 1000, 0)
# error_successive_over_relaxation = successive_over_relaxation(A, b, x, 1.25, 1000, 0)
#
# error_jacobi = ["Jacobi", list(range(1, len(error_jacobi))), error_jacobi]
# error_gauss_seidel = ["Gauss seidel", list(range(1, len(error_gauss_seidel))), error_gauss_seidel]
# error_successive_over_relaxation = ['Successive Over relaxation', list(range(1, len(error_successive_over_relaxation))), error_successive_over_relaxation]
# data_points = [error_jacobi, error_gauss_seidel, error_successive_over_relaxation]
#
#
# show_plot("", "", "", data_points)
# x = list(range(0, error.shape[0]))
# y = list(error)
# show_plot("", "", "", x, y, x, y)

def f(x):
    A = np.array([
        [1, 1, 1, 1, 76],
        [8, 4, 2, 1, 53],
        [64, 16, 4, 1, 73],
        [125, 25, 5, 1, 69]
    ], dtype=float)
    solution = simple_gaussian_elimination_method(A)
    return (solution[0] * (x ** 3)) + (solution[1] * (x ** 2)) + (solution[2] * (x)) + (solution[3])


vfunc = np.vectorize(f)

x_ = np.linspace(1, 6, 100)
result = vfunc(np.array(x_))

print("")



A = np.array([
    [1, 1, 1, 1],
    [8, 4, 2, 1],
    [64, 16, 4, 1],
    [125, 25, 5, 1]
], dtype=float)
b = np.array([76, 53, 73, 69], dtype=float)
x = np.array([
    [0, 0, 0, 0]
])

x = jacobi_method(A, b, x, 1000, 0.0005)
# x = simple_gaussian_elimination_method(A)
print("")
