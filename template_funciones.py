import numpy as np
import scipy
from scipy.linalg import solve_triangular


def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

 
def calculaLU(matriz: np.ndarray) -> tuple[np.ndarray]: # Es la que arrancamos en clase, la saqué del ejercicio 3 porque es probable que la usemos en varios lados
    """
    Recorre la matriz con  tres indices: k para la diagonal principal,
    i para las filas, j para las columnas. Replica el algoritmo para
    tringular pero en el lugar donde irian los ceros almacena los
    factores de eliminación gaussiana.
    Nos permite resolver (si es posible) el sistema Ax = b haciendo:
    1. L, U = calculaLU(A)
    2. y = solve_triangular(L, b, lower=True)
    3. x = solve_triangular(U, y) 
    """
    m = matriz.shape[0]
    n = matriz.shape[1]
    
    # Si no la copiamos como float da errores raros en algunos coeficientes, 
    # el que nos pasaron da bien sin la conversión, pero para otros casos nos trunca el cociente
    # y arrastra errores
    Ac = matriz.copy().astype(float)  
    if m != n:
        print('Matriz no cuadrada')
        return

    k = -1  
    while k < m-1: # Avance en diagonal
        k += 1 
        i = k 
        while i < m-1:   # Avance por filas         
            i += 1                       
            c = Ac[i][k] / Ac[k][k] # Factor de eliminación 
            j = k  # igualo a k para que empiece abajo de la diagonal (despues de la primera vuelta ya registré el factor de eliminación)
            while j < n: # Avance por columnas
                Ac[i][j] = Ac[i][j] - (c * Ac[k][j])
                j+=1                
            Ac[i][k] = c  # Registro en Ac el factor de eliminación después de usarlo en cada una de las j columnas
            
    L = np.tril(Ac,-1) + np.eye(matriz.shape[0]) 
    U = np.triu(Ac)
    
    return L, U


def calcula_inversa_con_LU(M: np.ndarray) -> np.ndarray: 
    """
    Recibe una matriz M invertible y retorna su inversa, utilizando factorización LU.
    """
    # Verificamos que M sea cuadrada
    filas = M.shape[0]
    columnas = M.shape[1]
    if filas != columnas:
        raise ValueError('Matriz no cuadrada')

    # Verificamos que M sea invertible: podríamos ver si el número de condición es muy grande
    # o si el rango de la matriz es menor estricto que n (dimension de filas o columnas), la matriz
    # sería singular, pero no se si nos está permitido usarlo. El determinante calculo que sí.
    # Otra opción seria pedir que M sea invertible
    det_M = np.linalg.det(M)
    if np.isclose(det_M, 0):  # Si det(M) != 0, M y LU son invertibles        
        raise ValueError('Matriz no invertible')

    # Factorizamos M
    L, U = calculaLU(M)
    Minv = np.zeros((filas, columnas))  # Creamos una matriz nula que contendrá a los x_j en sus columnas
    for j in range(columnas):       
        e_j = np.zeros(filas)  # Creamos el canónico e_j
        e_j[j] = 1
        y_j = solve_triangular(L, e_j, lower=True)  # Solución de Ly_j = e_j 
        x_j = solve_triangular(U, y_j) # Solución de Ux_j = y_j         
        Minv[:, j] = x_j  # Ubicamos el vector x_j como columna j de la inversa
    return Minv


def calcula_matriz_C(A: np.ndarray) -> np.ndarray:
    """
    Calcula y retorna la matriz de trancisiones C, a partir de una matriz de adyacencia A.
    """
    sumas_filas_A = np.array([np.sum(A[i][:]) for i in range(A.shape[0])])
    inversos_de_las_sumas = 1 / sumas_filas_A 
    # K = np.diag(sumas_filas_A)  
    Kinv = np.diag(inversos_de_las_sumas) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    # print(K @ Kinv, Kinv @ K)  # Para chequear si da bien
    C = A.T @ Kinv  # Calcula C multiplicando Kinv y A (Usamos fórmula (2) dada en el TP)
    return C

    
def calcula_pagerank(A: np.ndarray, alfa:float) -> np.ndarray:
    """
    Recibe una matriz de adyacencia A y un factor de amortiguamiento alfa
    (tambien llamado coeficiente de damping d), calcula la matriz de transiciones C
    y utilizando la descomposición LU para resolver Mp = b, retorna el vector p
    con los coeficientes de page rank de cada museo.
    """   
    C = calcula_matriz_C(A)
    N = A.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    # Usamos el M notado en el Ejercicio 1, pero sin el coeficiente N/α porque su inverso multiplicará a b (ver ejercicio 1)
    # Entonces: M = (I - (1 - α)C)
    I = np.eye(A.shape[0], A.shape[1])
    M = I - ((1 - alfa)*C)
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d (alfa)      
    b = (alfa/N) * np.ones(N) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d (alfa) y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p


def calcula_matriz_C_continua(D: np.ndarray) -> np.ndarray:
    """
    Recibe una matriz de distancias y calcula una matriz de transiciones para una red pesada.
    """ 
    # Función para calcular la matriz de trancisiones C
    # Retorna la matriz C en versión continua (red pesada)
    D = D.copy() # Matriz de distancias
    D_sin_cero = np.where(D == 0, np.nan, D)  # Cuando la distancia es cero reemplazamos con np.nan para no dividir entre 0 y evitar la advertencia
    # F = 1/D    
    F = 1/D_sin_cero # Es como la matriz de adyacencia A, pero pesada: está en términos de la función f(d_ji)  
    np.fill_diagonal(F,0)
    sumas_filas_F = np.array([np.sum(F[i][:]) for i in range(F.shape[0])])
    # K = np.diag(sumas_filas_F)      
    inversos_de_las_sumas = 1 / sumas_filas_F     
    Kinv = np.diag(inversos_de_las_sumas) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    # print(K @ Kinv, Kinv @ K)  # Para chequear si da bien       
    C = F.T @ Kinv # Calcula C multiplicando Kinv y F
    
    return C


def calcula_B(C: np.ndarray, cantidad_de_visitas:int) -> np.ndarray:
    """
    Recibe una matriz de transiciones y una cantidad de pasos, y devuelve
    la suma de las transiciones desde el inicio hasta la cantidad de pasos indicados. 
    """
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
  
    B = np.eye(C.shape[0])  # Es como tener B = C^0I
    potencias_de_C = C.copy()  # Acumula los productos sin modificar la matriz original
    for i in range(cantidad_de_visitas-1):
        # Sumamos las matrices de transición para cada cantidad de pasos
        B += potencias_de_C  # Empieza como B = I + C^1 y en cada iteración i suma: I + C^1 + ... + C^{i+1} 
        potencias_de_C = C @ potencias_de_C # Aumenta en uno la potencia 

    return B