import numpy as np
from scipy.linalg import lu

class App:
    def __init__(self, matriz, vector):
        self.matriz = np.array(matriz, dtype=float)
        self.vector = np.array(vector, dtype=float)
        self.dimension = len(vector)
    
    def resolver_por_lu(self):
        try:
            P, L, U = self.factorizacion_lu()
            y = np.linalg.solve(L, np.dot(P, self.vector))
            x = np.linalg.solve(U, y)
            return x
        except np.linalg.LinAlgError:
            return "La factorización LU no es posible para esta matriz."
    
    def factorizacion_lu(self):
        P, L, U = lu(self.matriz)
        return P, L, U

    def resolver_por_jacobi(self, tol=1e-10, max_iter=1000):
        D = np.diag(self.matriz)
        if np.any(D == 0):
            return "El método de Jacobi no es aplicable porque la matriz tiene ceros en la diagonal."
        R = self.matriz - np.diagflat(D)
        x = np.zeros_like(self.vector)
        for _ in range(max_iter):
            x_nuevo = (self.vector - np.dot(R, x)) / D
            if np.linalg.norm(x_nuevo - x, ord=np.inf) < tol:
                return x_nuevo
            x = x_nuevo
        return "El método de Jacobi no convergió."
    
    def resolver_por_gauss_jordan(self):
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
        print("Sistema de ecuaciones:")
        for i in range(self.dimension):
            ecuacion = " + ".join(f"{self.matriz[i, j]}x{j+1}" for j in range(self.dimension))
            print(f"{ecuacion} = {self.vector[i]}")

def main():
    n = int(input("Ingrese el tamaño del sistema (2, 3 o 4): "))
    
    if n not in [2, 3, 4]:
        print("❌ Solo se permite resolver sistemas de tamaño 2x2, 3x3 o 4x4.")
        return

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

    print("\n------------------------")
    sistema.imprimir_sistema()
    print("\nSolución por LU:", sistema.resolver_por_lu())
    print("Solución por Jacobi:", sistema.resolver_por_jacobi())
    print("Solución por Gauss-Jordan:", sistema.resolver_por_gauss_jordan())
    print("------------------------\n")

if __name__ == "__main__":
    main()
