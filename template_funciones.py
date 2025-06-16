import geopandas as gpd # Para hacer cosas geográficas
import matplotlib.pyplot as plt
import networkx as nx # Construcción de la red en NetworkX
import numpy as np
import pandas as pd # Para leer archivos
import random
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


def calcula_inversa_con_LU(M: np.ndarray, ver_det=True) -> np.ndarray: 
    """
    Recibe una matriz M invertible y retorna su inversa, utilizando factorización LU.
    """
    # Verificamos que M sea cuadrada
    filas = M.shape[0]
    columnas = M.shape[1]
    if filas != columnas:
        raise ValueError('Matriz no cuadrada')

    # Agregamos la posibilidad de no verificar que el determinante sea
    # cercano a cero, para matrices que están cerca de ser singulares
    # (Modificación necesaria para segunda parte del TP)
    if ver_det: 
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






def graficar_red_museos(
    m: int,
    alfa: int | float,
    n_museos_principales: int = 3,
    color_fondo: str = '#2a3fff',  # Acá podemos jugar con los parámetros para darle estilo a los gráficos
    color_barrio_relevante: str = '#7000c9',
    color_limite_barrial: str = '#ff9187',
    color_nodos_relevantes: str = '#ff0b69',
    color_nodos_grales: str = '#31ffba',
    color_texto: str = 'white',
    ax=None,  # Cuando se crean varios graficos seguidos le pasamos el eje como argumento
    fig=None  # Cuando se crean varios graficos seguidos le pasamos la figura como argumento 
    ):
    """
    Función que recibe una cantidad m de museos cercanos y un factor de amortiguamiento alfa
    y los utiliza para construir la matriz de adyacencia y de transiciones que permiten obtener
    el vector PageRank y utilizarlo para crear y graficar una red de museos ubicados en la Capital Federal.
    Además permite configurar algunos parámetros visuales del gráfico.
    """

    # Verificamos que los argumentos recibidos sean adecuados
    if m < 1:
        raise ValueError('La cantidad de museos cercanos debe ser mayor o igual a 1')
    if not isinstance(m, int):
        raise ValueError('El parámetro m debe ser un entero')
    if alfa < 0 or alfa > 1:
        raise ValueError('El parametro alfa debe estar en el intervalo [0, 1]')

    # Datos de los museos y barrios
    museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
    barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
    # Matriz de distancias
    D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy() 
    A = construye_adyacencia(D,m)  # Matriz de adyacencia
    p = calcula_pagerank(A, alfa)  # Vector de PageRank
    G = nx.from_numpy_array(A)  # Grafo (red) a partir de la matriz de adyacencia
    G_layout = {  # Diseño del grafo a partir de las coordenadas geográficas
        i:v for i,v in enumerate(
            zip(
                museos.to_crs("EPSG:22184").get_coordinates()['x'],
                museos.to_crs("EPSG:22184").get_coordinates()['y']
               )
        )
    }
    
    Nprincipales = n_museos_principales  # Cantidad a considerar de museos principales
    principales = np.argsort(p)[-Nprincipales:]
    labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Indices de los n museos principales

    # Obtenemos los nombres de los museos
    nombres_museos = []
    for indice_museo in principales:
        nombres_museos.append(museos['name'].iloc[indice_museo])

    # Obtenemos los barrios a colorear
    barrios_con_museos_relevantes = []
    for indice_museo in principales:
        museo = museos.iloc[indice_museo].geometry  # Obtenemos la ubicación
        # Obtenemos el barrio que contiene al museo
        barrios_con_museos_relevantes.append(barrios[barrios.geometry.contains(museo)]['nombre'].values[0])  
    barrios_a_colorear = list(set(barrios_con_museos_relevantes))  # Filtramos repetidos

    # Creamos un arreglo de booleanos que indica si se debe colorear o no cada barrio 
    aplicar_color = barrios['nombre'].isin(barrios_a_colorear) 

    # Visualización
    factor_escala = 1e4

    # Si no nos pasaron un eje, creamos figura y eje nuevos
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        nuevo_fig_creado = True
    else:
        nuevo_fig_creado = False

    # Configuración del eje
    ax.axis("off")
    if fig is not None:
        fig.set_facecolor(color_fondo)
    elif nuevo_fig_creado:
        fig.set_facecolor(color_fondo)

    # Al parecer, boundary.pot(...) solo hace los bordes pero no permite el relleno, asique usamos plot directamente
    # para colorear a los barrios que contienen a los n museos principales
    barrios.to_crs("EPSG:22184").plot(
        ax = ax,
        facecolor = np.where(aplicar_color, color_barrio_relevante, 'none'),  
        edgecolor = color_limite_barrial,
        linewidth = 2, # Grosor del límite de cada barrio
    )

    # Le damos un color diferente a los nodos principales, para que se distingan
    colores_de_nodos = [  
        color_nodos_relevantes if i in principales else color_nodos_grales for i in range(len(G.nodes))
    ]

    # Graficamos la red donde cada nodo tiene un tamaño proporcional al PageRank que le toco
    nx.draw_networkx(
        G,  # La red 
        G_layout,  # Su diseño sobre el mapa de CABA
        node_size = p*factor_escala,  # Array que asigna a cada nodo el tamaño correspondiente
        ax = ax,
        with_labels = False,
        node_color = colores_de_nodos)
    _ = nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k")  # Asignamos el retorno a una variable desechable para que no lo imprima 

    # Título y pie de gráfico
    titulo = f'RED DE MUSEOS CON PAGERANK Y UBICACIÓN DE LOS {n_museos_principales} PRINCIPALES \n(m = {m}, α = {np.round(alfa, 3)})\n\n'
    ax.set_title(
        titulo, 
        fontsize=9,
        color=color_texto,
        fontweight='bold',
    )

    # Generamos el texto del pie de gráfico
    nombres_principales = '\n '.join(str(nombre) for nombre in nombres_museos)
    tex_nombres = f'Los {Nprincipales} museos principales son:\n{nombres_principales}\n'
    barrios_principales = ', '.join(str(barrio) for barrio in barrios_a_colorear)
    tex_barrios = f'Barrio(s) con los {Nprincipales} museos principales: {barrios_principales}.'    
    info = tex_nombres + tex_barrios
    _ = fig.text(  # Asignamos el retorno a una variable desechable para que no lo imprima 
        0.65,
        0.1, 
        info,
        ha='center',  
        fontsize=9,
        color=color_texto,
    )
    
    if nuevo_fig_creado:
        plt.close()
        return fig
    else:
        return ax  # Devolvemos el eje modificado







# FUNCIONES DE LA SEGUNDA PARTE DEL TP

# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])

s_esperada = np.array([1, 1, 1, 1, -1, -1, -1, -1]) # 2 Grupos nodos del 1 al 4, y nodos del 5 al 8


def calcula_L(A: np.ndarray) -> np.ndarray:
    """
    La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    """
    # Sumamos las filas de A y almacenamos el resultado en un vector
    sumas_filas_A = np.array([np.sum(A[i, :]) for i in range(A.shape[0])])
    # Definimos el vector como la diagonal de K
    K = np.diag(sumas_filas_A) 
    # Retornamos L = K - A
    return K - A  

### Verificamos que L cumple con las propiedades que debe tener
##res_L = calcula_L(A_ejemplo)
##print(f'L es simétrica: {np.allclose(res_L, res_L.T)}')
##x_prueba = 2 * np.random.random(res_L.shape[0]) - 1
##print(f'Es semidefinida positiva: {x_prueba @ res_L @ x_prueba.T >= 0}')






def calcula_R(A:np.ndarray) -> np.ndarray:
    """
    La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    """
    # Construimos la diagonal de K, ya que solo necesitamos sus elementos diagonales
    diagonal_K = np.array([np.sum(A[i, :]) for i in range(A.shape[0])])
    # Calculamos el número total de conexiones E
    E = np.sum(A) / 2
    # Calculamos la matriz P que contiene el número esperado de conexiones entre i y j 
    P = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            P[i, j] = (diagonal_K[i]*diagonal_K[j]) / (2*E)
    # Retornamos R = A - P
    return A - P

### Verificamos que R cumple con las propiedades que debe tener
##res_R = calcula_R(A_ejemplo)
##print(f'R es simétrica: {np.allclose(res_R, res_R.T)}')
##sum_filas = np.isclose(0, np.sum(np.array([np.sum(res_R[i, :]) for i in range(res_R.shape[0])])))
##sum_cols = np.isclose(0, np.sum(np.array([np.sum(res_R[:, j]) for j in range(res_R.shape[1])])))
##print(f'La suma de las filas de R es cero: {sum_filas}')
##print(f'La suma de las columnas de R es cero: {sum_cols}')





def calcula_lambda(L,v):
    """
    Recibe L y v y retorna el corte asociado
    """
    # Definimos s
    s = np.where(np.isclose(v, 0) | (v > 0), 1, -1)
    # Calculamos el corte
    lambdon = (1/4) * (s.T @ L @ s)
    
    return lambdon



def calcula_Q(R,v):
    """
    La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    """
    # Definimos s
    s = np.where(np.isclose(v, 0) | (v > 0), 1, -1)
    # Calculamos la modularidad: la fórmula que nos pasaron incluye (1/4E)
    # pero para calcular E necesitamos A, que no lo tenemos, entonces devolvemos
    # Q' al cual se debería multiplicar por (1/2E) antes de usarlo
    Q = (1/2) * (s.T @ R @ s)
    return Q


### Verificamos con la partición esperada (s optimo)
##res_R = calcula_R(A_ejemplo)
##E = np.sum(A_ejemplo) / 2
##print(f'Modularidad con partición optima: {(1 / (2*E)) * calcula_Q(res_R, s_esperada)}') 





    

def metpot1(A:np.ndarray, tol:float=1e-8, maxrep:int=np.inf, semilla:int=1):
    """
    Recibe una matriz A y calcula su autovalor de mayor módulo,
    con un error relativo menor a tol y-o haciendo como mucho
    maxrep repeticiones.
    """
    np.random.seed(semilla) # Definimos una semilla para obtener resultados reproducibles
    
    # La fórmula para el cociente de Rayleigh nos pide dividir por: np.linalg.norm(v)**2
    # Pero probando varias veces, la calculada suele ser 0.9999... Entonces para arrastrar
    # la menor cantidad de errores posibles asumimos que np.linalg.norm(v) = 1 (lo quitamos del denominador)
    v = 2 * np.random.rand(A.shape[0]) - 1  # Generamos un vector de partida aleatorio, entre -1 y 1
    v /= np.linalg.norm(v)  # Lo normalizamos: asumimos que que np.linalg.norm(v) = 1
    v1 = A @ v  # Aplicamos la matriz una vez
    v1 /= np.linalg.norm(v1)  # normalizamos
    l = np.dot(v, A @ v)  # Calculamos el autovector estimado: np.linalg.norm(v)**2 = 1 (no lo agregamos)
    l1 = np.dot(v1, A @ v1)  # Y el estimado en el siguiente paso
    nrep = 0  # Contador
    prevencion = tol * 1e-2  # Evitamos la división por cero
    while np.abs(l) > prevencion and np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
        v = v1.copy()  # actualizamos v y repetimos
        l = l1
        v1 = A @ v1  # Calculo nuevo v1
        v1 /= np.linalg.norm(v1)  # Normalizo
        l1 = np.dot(v1, A @ v1)  # Calculo autovector
        nrep += 1  # Un pasito mas
    if not nrep < maxrep:
        print('MaxRep alcanzado')
    l = np.dot(v1, A @ v1)  # Calculamos el autovalor    
    return v1 ,l , nrep<maxrep

### Verificamos
##res = metpot1(A_ejemplo)
##print(f'El autovalor dominante de A es: {np.round(res[1], 8)}')




def deflaciona(
    A:np.ndarray, tol:float=1e-8, maxrep:int=np.inf, semilla:int=1
    ) -> np.ndarray:
    """
    Recibe la matriz A, una tolerancia para el método de la potencia,
    y un número máximo de repeticiones
    """
    v1, l1, _ = metpot1(A, tol, maxrep, semilla)  # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.outer(v1, v1)  # v1 ya está normalizado
    # Retornamos A_1 (el autovalor dominante obtenido fue reemplazado por 0)
    return deflA

### Verificamos
##A_1 = deflaciona(A_ejemplo)
##print(f'A_1 es simétrica: {np.allclose(A_1, A_1.T)}')
##print(f'El autovalor dominante de A_1 es: {np.round(metpot1(A_1)[1],8)}')





def metpot2(
    A:np.ndarray, v1:np.ndarray, l1:float, tol:float=1e-8, maxrep:int=np.inf, semilla:int=1
    ):
    """
    La funcion aplica el metodo de la potencia para buscar
    el segundo autovalor de A, suponiendo que sus autovectores
    son ortogonales.
    """
    # v1 y l1 son los primeros autovectores y autovalores de A}    
    deflA = A - l1 * np.outer(v1, v1)  # Esta es nuestra A_1 donde el autovalor dominante de A ahora es 0 
    return metpot1(deflA, tol, maxrep, semilla)

### Verificamos
##A_1 = deflaciona(A_ejemplo)
##v1, l1, _ = metpot1(A_ejemplo)
##v2, l2, _ = metpot2(A_ejemplo, v1, l1)
##print(f'El autovalor l1 es: {np.round(l1, 8)}\n')
##print(f'El autovalor l2 es: {np.round(l2, 8)}\n')





# La matriz laplaciana tiene autovalores >= 0, cuando mu ≈ 0 obtiene el buscado
# Para el corte mínimo usamos el segundo autovalor más chico
def metpotI(A:np.ndarray, mu:float, tol:float=1e-8, maxrep:int=np.inf, semilla:int=1): 
    """
    Retorna el primer autovalor de la inversa de A + mu * I,
    junto a su autovector y si el método convergió.
    """
    A_shifteada = A + mu * np.eye(A.shape[0])
    A_shifteada_inv = calcula_inversa_con_LU(A_shifteada, False)  # Evitamos que calcule el determinante 
    # Cuando retorna, el autovalor posta lo recuperamos con: λ = (1 / res[1]) - mu 
    return metpot1(A_shifteada_inv, tol, maxrep, semilla)

### Verificamos
##mu = 1.9  # No puede ser exactamente igual a un autovalor sino la matriz se vuelve singular 
##res = metpotI(A_ejemplo, mu)
##autoval_verdadero = (1 / res[1]) - mu
##print(autoval_verdadero)









def metpotI2(A:np.ndarray, mu:float, tol:float=1e-8, maxrep:int=np.inf, semilla:int=1):
    """
    Recibe la matriz A, y un valor mu y retorna el segundo autovalor
    y el autovector asociado de la matriz A, suponiendo que sus autovalores
    son positivos excepto por el menor que es igual a 0
    """
    X = A + mu * np.eye(A.shape[0])  # Calculamos la matriz A shifteada en mu
    iX = calcula_inversa_con_LU(X, False)  # La invertimos    
    defliX = deflaciona(iX, tol, maxrep, semilla)  # La deflacionamos
    v,l,_ =  metpot1(defliX, tol, maxrep, semilla)  # Buscamos su segundo autovector
    l = 1/l  # Reobtenemos el autovalor correcto
    l -= mu
    return v, l 

### Verificamos
##D = np.diag([7.0, 4.0, 2.0, 5.0, 1.0, 0.0, 8.0])  # Estos son los autovalores 
##v = np.random.randn(D.shape[0],1)
##v /= np.linalg.norm(v)
##B = np.eye(D.shape[0]) - 2 * v @ v.T
##A = B @ D @ B.T
##mu = 0.1
##res = metpotI2(A, mu)
##print(f'Segundo autovalor mas chico: {res[1]}\n')



def laplaciano_iterativo(
    A:np.ndarray, 
    niveles:int, 
    nombres_s=None, 
    mu:float=1e-4,  # Argumentos para llamar al método de la potencia inversa
    tol:float=1e-8, 
    maxrep:int=np.inf, 
    semilla:int=1
    ):
    """
    Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes,
    y los nombres de los nodos. Retorna una lista con conjuntos de nodos
    representando las comunidades. La función debe, recursivamente, ir realizando
    cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    """
    # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
    if nombres_s is None:
        nombres_s = range(A.shape[0])
    # Si llegamos al último paso, retornamos los nombres en una lista
    if A.shape[0] == 1 or niveles == 0: 
        return [nombres_s] 
    else: # Sino:
        L = calcula_L(A)  # Recalculamos el L
        # Definimos µ con un valor cercano a 0
        # mu = 1e-4  # Si otras matrices tienen autovalores más cercanos que cero a mu, hay que cambiarlo 
        v, _ = metpotI2(L, mu, tol, maxrep, semilla)  # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v
        # y la que está asociada al negativo:
       
        s = np.where(np.isclose(v, 0) | (v > 0), 1, -1)  # Calculamos el vector de asignación de comunidades      
        
        indices_nodos_positivos = np.where(s == 1)[0]  # Obtenemos los índices del grupo1 (s_i = 1)
        indices_nodos_negativos = np.where(s == -1)[0]  # Obtenemos los índices del grupo2 (s_i = -1)

        # if not grupo_pos or not grupo_neg:
        # return [nombres_s]
        
        Ap = A[indices_nodos_positivos][:, indices_nodos_positivos] # Grupo asociado al signo positivo       
        Am = A[indices_nodos_negativos][:, indices_nodos_negativos] # Grupo asociado al signo negativo

        return(
            laplaciano_iterativo(
                Ap,  # Se calcula la asignación de comunidades dentro del grupo1
                niveles-1,
                nombres_s=[ni for ni,vi in zip(nombres_s,v) if (np.isclose(vi, 0) or (vi > 0))]  # Lista de nodos en grupo1
            ) +
            laplaciano_iterativo(
                Am, # Se calcula la asignación de comunidades dentro del grupo2
                niveles-1,
                nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0] # Lista de nodos en grupo2
            ) 
        ) 
             
### Verificamos
##grupos = laplaciano_iterativo(A_ejemplo, 2)
##print(grupos) # Dividió en dos grupos la matriz de ejemplo




def modularidad_iterativo(
    A=None,
    R=None,
    nombres_s=None,
    tol:float=1e-8, # Argumentos para llamar al método de la potencia
    maxrep:int=np.inf, 
    semilla:int=1,
    ):
    """
    Recibe una matriz A, una matriz R de modularidad, y los nombres
    de los nodos. Retorna una lista con conjuntos de nodos representando
    las comunidades.
    """
    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None: # Solo se ejecuta en la primera llamada
        nombres_s = range(R.shape[0])
    
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return [nombres_s]
    
    else: # Si la matriz de modularidad R tiene dimensión mayor a 1        
        v,l, _ = metpot1(R, tol, maxrep, semilla) # Obtenemos el primer autovector y autovalor de R
        # Si E = np.sum(A) / 2, y tengo que hacer: (1 / (2*E)), me queda: 
        Q0 = (1 / np.sum(A)) * calcula_Q(R, v)  # Calculamos la modularidad actual
        
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return [nombres_s]

        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            v2, l2, _ = metpot2(R, v, l, tol, maxrep, semilla)  # Encontramos el segundo autovector de R            
            s = np.where(np.isclose(v2, 0) | (v2 > 0), 1, -1)  # Calculamos el vector de asignación de comunidades
            
            # La longitud de s se va achicando, por lo tanto el rango de los indices tambien
            indices_nodos_positivos = np.where(s == 1)[0]  # Indices grupo 1
            indices_nodos_negativos = np.where(s == -1)[0]  # Indices grupo -1
            
            Rp = R[indices_nodos_positivos][:, indices_nodos_positivos]  # Parte de R asociada a los valores positivos de v
            Rm = R[indices_nodos_negativos][:, indices_nodos_negativos]  # Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp, tol, maxrep, semilla)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm, tol, maxrep, semilla) # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
                Q1 = (1 / np.sum(A)) * calcula_Q(Rp, vp)
            if not all(vm>0) or all(vm<0):
                Q1 += (1 / np.sum(A)) * calcula_Q(Rm, vm)
            
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos                
                return [[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]]            
            else:
                # Sino, repetimos para los subniveles
                return(
                    modularidad_iterativo(
                        A[indices_nodos_positivos][:, indices_nodos_positivos], # Grupo asociado al signo positivo
                        Rp,
                        [nombres_s[i] for i in indices_nodos_positivos]  # Lo indices son relativos a un R de tamaño distinto del original
                    ) +
                    modularidad_iterativo(
                        A[indices_nodos_negativos][:, indices_nodos_negativos], # Grupo asociado al signo negativo
                        Rm,
                        [nombres_s[i] for i in indices_nodos_negativos]
                    )                  
                )

### Verificamos
##grupos = modularidad_iterativo(A_ejemplo, calcula_R(A_ejemplo))
##print(grupos) # Dividió en dos grupos la matriz de ejemplo





# FUNCIONES DEL PUNTO 4 (Museos y comunidades)

def construye_adyacencia_simetrica(D:np.ndarray, m:int) -> np.ndarray:
    """
    Recibe una matriz de distancias y una cantidad m de museos cercanos
    y retorna la matriz de adyacencia simetrizada.
    """    
    # Condición sobre el argumento
    if m < 1 or not isinstance(m, int):
        raise ValueError('m debe ser un entero positivo')        

    A = construye_adyacencia(D, m)
    
    return np.ceil(0.5 * (A + A.T))



colores = [  # Para identifcar las comunidades (se eligen al azar)
    "#004d03","#5f33de","#51bf00","#a754ff","#00be3b","#ff4ef2","#85bd00","#48009e","#caad00","#3764ff",
    "#e79b00","#0129af","#f37200","#3095ff","#ff2d1d","#00a9fe","#ea0020","#00bd8d","#fb00c5","#00ae68",
    "#9d0095","#a9cd75","#ff54c7","#385d00","#7489ff","#927e00","#d190ff","#666e00","#730066","#ccc46b",
    "#001b52","#ff113f","#01b4b0","#ff5141","#01adcd","#be4300","#0059a4","#ff884a","#0182b6","#9a1400",
    "#59d4ca","#b20067","#007e54","#ff6285","#003a0b","#ff7eb2","#016f54","#ff7e79","#001804","#ff9ed6",
    "#004839","#ffa67d","#180025","#e1bc85","#59003d","#b2c8a7","#830039","#006467","#ad5a00","#016295",
    "#7c5300","#bcbcfd","#622000","#bac1d9","#330300","#e1b5d1","#3e2e00","#d3bdb9","#004052","#f4b29d",
    "#b60004","#05bc3c","#8e11a8","#cbc000","#2050da","#99ac00","#7535c1","#b6cc52","#ab6dfe","#007c1c",
    "#e24cd4","#01a85a","#da009b","#3ada9a","#aa008d","#548100","#3f7dff","#ffa825","#0045b2","#f07100",
    "#014eac","#b29000","#573395","#9fd073","#c4005a","#6dd5a2","#ff3948","#00c4dd","#d5380a","#02bafd",
    "#9e1c00","#01a5a2","#bf0036","#018d67","#ff6096","#005719","#ffa2f4","#4b5800","#b49dff","#cb7100",
    "#015998","#bb7c00","#60367d","#95d187","#782c5c","#e6bc5d","#414479","#ff9652","#01566f","#a65300",
    "#b4b8ff","#873600","#8fceb9","#8a2115","#acc9b2","#7d2f29","#dabe87","#3e4863","#ff7f74","#365027",
    "#ff8bc3","#814e00","#f3add5","#614226","#ddb7cb","#594c40","#ffa187","#5e404c","#f5b0b0","#98827d"
]

def obtener_colores_nodos(lista_comunidades:list[list]) -> list[str]:
    """
    Recibe una lista de comunidades y le asigna el mismo color a sus miembros.
    """
    lista_colores = ['']*136
    for comunidad in lista_comunidades:
        color = np.random.randint(139)
        for indice_nodo in comunidad:
            lista_colores[indice_nodo] = colores[color]
    return lista_colores        



def graficar_comunidades_museos_laplaciano(
    m: int,
    mu:float=1e-4,
    corte_minimo=None,
    tol:float=1e-8, 
    maxrep:int=np.inf,
    semilla:int=1,    
    color_fondo: str = '#fce7bc',  # Acá podemos jugar con los parámetros para darle estilo a los gráficos
    color_barrio_relevante: str = '#d5c39f',
    color_limite_barrial: str = '#a39579',
    color_texto: str = 'black',
    ax=None,  # Cuando se crean varios graficos seguidos le pasamos el eje como argumento
    fig=None  # Cuando se crean varios graficos seguidos le pasamos la figura como argumento 
    ):
    """
    Función que recibe una cantidad m de museos cercanos y un valor mu cercano al autovalor buscado
    y los utiliza para construir la matriz de adyacencia simétrica y detectar comunidades.
    Además permite configurar algunos parámetros visuales del gráfico.
    """

    # Verificamos que los argumentos recibidos sean adecuados
    if m < 1:
        raise ValueError('La cantidad de museos cercanos debe ser mayor o igual a 1')
    if not isinstance(m, int):
        raise ValueError('El parámetro m debe ser un entero positivo')

    # Datos de los museos y barrios
    museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
    barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
    
    # Matriz de distancias
    D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy() 
    
    A = construye_adyacencia_simetrica(D, m)  # Matriz de adyacencia

    # Detectamos las comunidades
    L = calcula_L(A)  # Calculamos la matriz laplaciana
    res_L = metpotI2(L, mu, tol=tol, maxrep=maxrep, semilla=semilla)  # Obtenemos el segundo autovector mas chico
    s_calculado= np.where(np.isclose(res_L[0], 0) | (res_L[0] > 0), 1, -1)  # Calculamos el vector de asignación 

    # Obtenemos la lista de comunidades
    if corte_minimo:
        cm = corte_minimo # Le pasamos un corte arbitrario
    else:
        cm = calcula_lambda(L, s_calculado)  # Calculamos el corte mínimo con la función que definimos
    particion_con_L = laplaciano_iterativo(A, int(cm), tol=tol, maxrep=maxrep, semilla=semilla)  

    G_comunidades = nx.Graph() # Creamos un grafo vacío 
    G_comunidades.add_nodes_from(range(A.shape[0]))  # Misma cantidad de nodos que la matriz de adyacencia simétrica

    # Solo dibujamos las aristas entre museos de una misma comunidad
    for grupo in particion_con_L:
        for i in range(len(grupo)):
            for j in range(i + 1, len(grupo)):
                u, v = grupo[i], grupo[j]
                if A[u, v] != 0:  # Si hay una conexión en la matriz original
                    G_comunidades.add_edge(u, v)
    

    grafo_geo_comunidades = {  # Diseño del grafo a partir de las coordenadas geográficas
        i:v for i,v in enumerate(
            zip(
                museos.to_crs("EPSG:22184").get_coordinates()['x'],
                museos.to_crs("EPSG:22184").get_coordinates()['y']
               )
        )
    }
    
    cant_museos = list(range(A.shape[0]))

    # Obtenemos los barrios a colorear
    barrios_con_museos = []
    for indice_museo in cant_museos:
        museo = museos.iloc[indice_museo].geometry  # Obtenemos la ubicación
        # Obtenemos el barrio que contiene al museo
        barrio = barrios[barrios.geometry.contains(museo)]
        try:
            barrios_con_museos.append(barrio['nombre'].values[0])  
        except:  # Si está entre dos barrios
            entre_barrios = barrios[barrios.geometry.intersects(museo)] 
            if not entre_barrios.empty:
                barrios_con_museos.append(entre_barrios['nombre'].values[0])            
    barrios_a_colorear = list(set(barrios_con_museos))  # Filtramos repetidos

    # Creamos un arreglo de booleanos que indica si se debe colorear o no cada barrio 
    aplicar_color = barrios['nombre'].isin(barrios_a_colorear) 

    # Visualización
    factor_escala = 1e4

    # Si no nos pasaron un eje, creamos figura y eje nuevos
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        nuevo_fig_creado = True
    else:
        nuevo_fig_creado = False

    # Configuración del eje
    ax.axis("off")
    if fig is not None:
        fig.set_facecolor(color_fondo)
    elif nuevo_fig_creado:
        fig.set_facecolor(color_fondo)

    # Al parecer, boundary.pot(...) solo hace los bordes pero no permite el relleno, asique usamos plot directamente
    # para colorear a los barrios que contienen a los n museos principales
    barrios.to_crs("EPSG:22184").plot(
        ax = ax,
        facecolor = np.where(aplicar_color, color_barrio_relevante, 'none'),  
        edgecolor = color_limite_barrial,
        linewidth = 2, # Grosor del límite de cada barrio
    )

    # Coloreamos los nodos según la comunidad a la que pretenecen
    colores_de_nodos = obtener_colores_nodos(particion_con_L)

    # Graficamos la red (todos los nodos tienen el mismo tamaño)
    nx.draw_networkx(
        G_comunidades, 
        grafo_geo_comunidades,  # El diseño de las comunidades sobre el mapa de CABA        
        # Era p antes de ser 0.01
        node_size = 0.015*factor_escala,  # Array que asigna a cada nodo el tamaño correspondiente
        ax = ax,
        with_labels = False,
        node_color = colores_de_nodos)
    # Eliminamos la instrcucción que imprime los nombres (índices) de los museos

    # Título y pie de gráfico
    titulo = f'COMUNIDADES DE MUSEOS CERCANOS\nm = {m}, corte mínimo = {cm}'
    ax.set_title(
        titulo, 
        fontsize=9,
        color=color_texto,
        fontweight='bold',
    )

    # Generamos el texto del pie de gráfico
    info = f'{len(particion_con_L)} comunidades detectadas con el laplaciano'
    _ = fig.text(  # Asignamos el retorno a una variable desechable para que no lo imprima 
        0.65,
        0.1, 
        info,
        ha='center',  
        fontsize=9,
        color=color_texto,
    )
    
    if nuevo_fig_creado:
        plt.close()
        return fig
    else:
        return ax  # Devolvemos el eje modificado


# Esta función hace practicamente lo mismo que la anterior en términos de digramación
# pero hace llamadas a métodos distintos (entre otras cosas) 
# Para agilizar la lectura del notebook, decidimos definirla por separado
def graficar_comunidades_museos_modularidad(
    m: int,
    tol:float=1e-8, 
    maxrep:int=np.inf,
    semilla:int=1,
    color_fondo: str = '#d4c0ff',  # Acá podemos jugar con los parámetros para darle estilo a los gráficos
    color_barrio_relevante: str = '#ac9ccf',
    color_limite_barrial: str = '#7d7196',
    color_texto: str = 'black',
    ax=None,  # Cuando se crean varios graficos seguidos le pasamos el eje como argumento
    fig=None,  # Cuando se crean varios graficos seguidos le pasamos la figura como argumento
    solo_m_aristas:bool=True  # Señal para indicar si queres conectar o no todos los nodos de cada comunidad 
    ):
    """
    Función que recibe una cantidad m de museos cercanos y lo utiliza para construir 
    la matriz de adyacencia simétrica y detectar comunidades.
    Además permite configurar algunos parámetros visuales del gráfico.
    """

    # Verificamos que los argumentos recibidos sean adecuados
    if m < 1:
        raise ValueError('La cantidad de museos cercanos debe ser mayor o igual a 1')
    if not isinstance(m, int):
        raise ValueError('El parámetro m debe ser un entero positivo')

    # Datos de los museos y barrios
    museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
    barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
    
    # Matriz de distancias
    D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy() 
    
    A = construye_adyacencia_simetrica(D, m)  # Matriz de adyacencia

    # Detectamos las comunidades    
    R = calcula_R(A)
    res_R = metpot1(R, tol=tol, maxrep=maxrep, semilla=semilla)
    particion_con_modularidad = modularidad_iterativo(A, R, tol=tol, maxrep=maxrep, semilla=semilla) # Vector de asignación con modularidad

    G_comunidades = nx.Graph() # Creamos un grafo vacío 
    G_comunidades.add_nodes_from(range(A.shape[0]))  # Misma cantidad de nodos que la matriz de adyacencia simétrica

    # Dibujamos las aristas entre museos de una misma comunidad
    for grupo in particion_con_modularidad:
        for i in range(len(grupo)):
            for j in range(i + 1, len(grupo)):
                u, v = grupo[i], grupo[j]
                if solo_m_aristas:  # Solo conectamos si hay una conexión en la matriz original
                    if A[u, v] != 0:  
                        G_comunidades.add_edge(u, v)
                else:  # Conectamos entre sí a todos los miembros de cada comunidad 
                    G_comunidades.add_edge(u, v)                  
    

    grafo_geo_comunidades = {  # Diseño del grafo a partir de las coordenadas geográficas
        i:v for i,v in enumerate(
            zip(
                museos.to_crs("EPSG:22184").get_coordinates()['x'],
                museos.to_crs("EPSG:22184").get_coordinates()['y']
               )
        )
    }
    
    cant_museos = list(range(A.shape[0]))

    # Obtenemos los barrios a colorear
    barrios_con_museos = []
    for indice_museo in cant_museos:
        museo = museos.iloc[indice_museo].geometry  # Obtenemos la ubicación
        # Obtenemos el barrio que contiene al museo
        barrio = barrios[barrios.geometry.contains(museo)]
        try:
            barrios_con_museos.append(barrio['nombre'].values[0])  
        except:  # Si está entre dos barrios
            entre_barrios = barrios[barrios.geometry.intersects(museo)] 
            if not entre_barrios.empty:
                barrios_con_museos.append(entre_barrios['nombre'].values[0])            
    barrios_a_colorear = list(set(barrios_con_museos))  # Filtramos repetidos

    # Creamos un arreglo de booleanos que indica si se debe colorear o no cada barrio 
    aplicar_color = barrios['nombre'].isin(barrios_a_colorear) 

    # Visualización
    factor_escala = 1e4

    # Si no nos pasaron un eje, creamos figura y eje nuevos
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        nuevo_fig_creado = True
    else:
        nuevo_fig_creado = False

    # Configuración del eje
    ax.axis("off")
    if fig is not None:
        fig.set_facecolor(color_fondo)
    elif nuevo_fig_creado:
        fig.set_facecolor(color_fondo)

    # Al parecer, boundary.pot(...) solo hace los bordes pero no permite el relleno, asique usamos plot directamente
    # para colorear a los barrios que contienen a los n museos principales
    barrios.to_crs("EPSG:22184").plot(
        ax = ax,
        facecolor = np.where(aplicar_color, color_barrio_relevante, 'none'),  
        edgecolor = color_limite_barrial,
        linewidth = 2, # Grosor del límite de cada barrio
    )

    # Coloreamos los nodos según la comunidad a la que pretenecen
    colores_de_nodos = obtener_colores_nodos(particion_con_modularidad)

    # Graficamos la red (todos los nodos tienen el mismo tamaño)
    nx.draw_networkx(
        G_comunidades, 
        grafo_geo_comunidades,  # El diseño de las comunidades sobre el mapa de CABA        
        # Era p antes de ser 0.01
        node_size = 0.015*factor_escala,  # Array que asigna a cada nodo el tamaño correspondiente
        ax = ax,
        with_labels = False,
        node_color = colores_de_nodos)
    # Eliminamos la instrucción que imprime los nombres (índices) de los museos

    # Título y pie de gráfico
    titulo = f'COMUNIDADES DE MUSEOS CERCANOS\nm = {m}'
    ax.set_title(
        titulo, 
        fontsize=9,
        color=color_texto,
        fontweight='bold',
    )

    # Generamos el texto del pie de gráfico
    info = f'{len(particion_con_modularidad)} comunidades detectadas con modularidad'
    _ = fig.text(  # Asignamos el retorno a una variable desechable para que no lo imprima 
        0.65,
        0.1, 
        info,
        ha='center',  
        fontsize=9,
        color=color_texto,
    )
    
    if nuevo_fig_creado:
        plt.close()
        return fig
    else:
        return ax  # Devolvemos el eje modificado

