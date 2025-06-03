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





# FUNCIONES DE LA SEGUNDA PARTE DEL TP


# Matriz A de ejemplo
#A_ejemplo = np.array([
#    [0, 1, 1, 1, 0, 0, 0, 0],
#    [1, 0, 1, 1, 0, 0, 0, 0],
#    [1, 1, 0, 1, 0, 1, 0, 0],
#    [1, 1, 1, 0, 1, 0, 0, 0],
#    [0, 0, 0, 1, 0, 1, 1, 1],
#    [0, 0, 1, 0, 1, 0, 1, 1],
#    [0, 0, 0, 0, 1, 1, 0, 1],
#    [0, 0, 0, 0, 1, 1, 1, 0]
#])


def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    # Have fun!!
    return L

def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    # Have fun!!
    return R

def calcula_lambda(L,v):
    # Recibe L y v y retorna el corte asociado
    # Have fun!
    return lambdon

def calcula_Q(R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    return Q

def metpot1(A,tol=1e-8,maxrep=np.Inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = ... # Generamos un vector de partida aleatorio, entre -1 y 1
   v = ... # Lo normalizamos
   v1 = ... # Aplicamos la matriz una vez
   v1 = ... # normalizamos
   l = ... # Calculamos el autovector estimado
   l1 = ... # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = ... # Calculo nuevo v1
      v1 = ... # Normalizo
      l1 = ... # Calculo autovector
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = ... # Calculamos el autovalor
   return v1,l,nrep<maxrep

def deflaciona(A,tol=1e-8,maxrep=np.Inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = ... # Sugerencia, usar la funcion outer de numpy
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.Inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   # Have fun!
   return metpot1(deflA,tol,maxrep)


def metpotI(A,mu,tol=1e-8,maxrep=np.Inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    return metpot1(...,tol=tol,maxrep=maxrep)

def metpotI2(A,mu,tol=1e-8,maxrep=np.Inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = ... # Calculamos la matriz A shifteada en mu
   iX = ... # La invertimos
   defliX = ... # La deflacionamos
   v,l,_ =  ... # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_


def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v,l,_ = ... # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = ... # Asociado al signo positivo
        Am = ... # Asociado al signo negativo
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return(...)
    else:
        v,l,_ = ... # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return(...)
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = ... # Parte de R asociada a los valores positivos de v
            Rm = ... # Parte asociada a los valores negativos de v
            vp,lp,_ = ...  # autovector principal de Rp
            vm,lm,_ = ... # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(...)



    return B