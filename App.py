import numpy as np
from scipy.linalg import lu

class App:
    def __init__(self, matriz, vector):
        self.matriz = np.array(matriz, dtype=float)
        self.vector = np.array(vector, dtype=float)
        self.dimension = len(vector)
    
    def resolver_por_lu(self):
        print("LU")
        try:
            P, L, U = self.factorizacion_lu()
            print("\nMatriz P (permutación):")
            print(P)
            if np.allclose(P, np.eye(self.dimension)):
                print("\nNo fue necesaria la matriz de permutación (P es la identidad).")
            else:
                print("\nSe utilizó la matriz de permutación P para poder realizar la factorización.")
            print("\nMatriz L (triangular inferior):")
            print(L)
            print("\nMatriz U (triangular superior):")
            print(U)

            # 🚨 Aplica la permutación a A y b antes de resolver
            A_permutada = P @ self.matriz
            b_permutada = P @ self.vector

            # Resolución del sistema LUx = Pb
            y = np.linalg.solve(L, b_permutada)
            x = np.linalg.solve(U, y)

            return x
        except np.linalg.LinAlgError:
            return "La factorización LU no es posible para esta matriz."

    def factorizacion_lu(self):
        P, L, U = lu(self.matriz)
        return P, L, U

    def resolver_por_jacobi(self, tol=1e-10, max_iter=1000):
        print("JACOBI")
        print("\nSistema original:")
        self.imprimir_sistema()

        def es_dominante_por_filas(m):
            for i in range(len(m)):
                suma_otros = np.sum(np.abs(m[i])) - np.abs(m[i][i])
                if np.abs(m[i][i]) <= suma_otros:
                    return False
            return True

        def intentar_hacer_dominante(matriz, vector):
            from itertools import permutations
            n = len(matriz)
            for perm in permutations(range(n)):
                matriz_perm = matriz[list(perm), :]
                if es_dominante_por_filas(matriz_perm):
                    return matriz_perm, vector[list(perm)]
            return None, None

        if not es_dominante_por_filas(self.matriz):
            print("La matriz NO es diagonalmente dominante.")
            nueva_A, nuevo_b = intentar_hacer_dominante(self.matriz, self.vector)
            if nueva_A is None:
                return "No se pudo hacer el sistema bien condicionado para resolverlo por Jacobi."
            else:
                print("\nSistema reordenado para ser bien condicionado:")
                self.matriz = nueva_A
                self.vector = nuevo_b
                self.imprimir_sistema()
        else:
            print("\nLa matriz es diagonalmente dominante. Se procede a resolver.")

        # Estas variables deben estar disponibles en ambos casos
        D = np.diag(self.matriz)
        R = self.matriz - np.diagflat(D)
        x = np.zeros_like(self.vector)

        for _ in range(max_iter):
            x_nuevo = (self.vector - np.dot(R, x)) / D
            if np.linalg.norm(x_nuevo - x, ord=np.inf) < tol:
                return x_nuevo
            x = x_nuevo

        return "El método de Jacobi no convergió."


    def resolver_por_gauss_jordan(self):
        print("GAUSS-JORDAN\n")
        try:
            augmented = np.hstack((self.matriz, self.vector.reshape(-1, 1)))
            filas, columnas = augmented.shape
            for i in range(filas):
                if augmented[i, i] == 0:
                    return "No se puede resolver por Gauss-Jordan debido a un pivote cero."
                augmented[i] = augmented[i] / augmented[i, i]
                for j in range(filas):
                    if i != j:
                        augmented[j] -= augmented[i] * augmented[j, i]
            return augmented[:, -1]
        except:
            return "No se pudo resolver por Gauss-Jordan."
    
    def imprimir_sistema(self):
        for i in range(self.dimension):
            ecuacion = " + ".join(f"{self.matriz[i, j]}x{j+1}" for j in range(self.dimension))
            print(f"{ecuacion} = {self.vector[i]}")
        print("")

def main():
    n = 0
    while True:
        entrada = input("Ingrese el tamaño del sistema (2, 3 o 4): ")
        if entrada.isdigit():
            n = int(entrada)
            if n in [2, 3, 4]:
                break
        print("\nSolo se permite ingresar 2, 3 o 4.\n")

    matriz = []
    vector = []

    print("\nIntroduce los valores de la matriz A:")
    for i in range(n):
        fila = []
        for j in range(n):
            valor = float(input(f"A[{i+1}][{j+1}]: "))
            fila.append(valor)
        matriz.append(fila)

    print("\nIntroduce los valores del vector b:")
    for i in range(n):
        valor = float(input(f"b[{i+1}]: "))
        vector.append(valor)

    sistema = App(matriz, vector)

    print("----------------------------------------------------------------------------------------")
    print("Sistema de ecuaciones:")
    sistema.imprimir_sistema()
    print("----------------------------------------------------------------------------------------")
    print("\nSOLUCIÓN POR LU:", sistema.resolver_por_lu())
    print("----------------------------------------------------------------------------------------")
    print("SOLUCIÓN POR JACOBI:", sistema.resolver_por_jacobi())
    print("----------------------------------------------------------------------------------------")
    print("SOLUCIÓN POR GAUSS-JORDAN:", sistema.resolver_por_gauss_jordan())
    print("----------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()

