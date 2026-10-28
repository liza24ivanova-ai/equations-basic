M = 80
DELTA = 0.025
ETAS = [1e-4, 1e-2]
LAMBDAS = [10 ** (-6 + k / 4) for k in range(25)]


def main():
    print(M, DELTA, ETAS, len(LAMBDAS))

    # your code here \/
    ...
    # your code here /\


if __name__ == "__main__":
    main()
