import math
import numpy as np
import matplotlib.pyplot as plt

#1 1,1,10,200       
#5 13,5,13,10

def phi_1(t): # x = 0
    #return 0
    #return 0
    #return 1
    #return 0
    #return 1 #- нелинейное
    #return 0  # Нужный пример
    #return t-1
    #return 0
    #return t
    return np.sqrt(4*t) # Локализация


def phi_2(t): #x = l
    #return 0
    #return t-1
    #return math.cos(l*t)
    #return np.sin(l)*np.exp(l/(t+3))
    #return np.exp(-t) #- нелинейное
    #return 0   # Нужный пример
    #return t
    #return 0
    #return t*t/2+t
    return 0 # Локализация

def u_0(x): # t = 0
    #return np.sin(np.pi/l * x)
    #return np.sin(3*np.pi*x/2)
    #return 1
    #return np.sin(x)*np.exp(x/3)
    #return 1 #- нелинейное
    #return np.sin(np.pi * x) # Нужный пример
    #return x*x-1
    #return 0
    #return 0
    return 0.0 # Локализация

def f(x, t, u = 0.0):
    #return 0
    #return 9*np.pi*np.pi/4*np.sin(3*np.pi*x/2)-2*t+x*x
    #return t*t*math.cos(x*t)-x*math.sin(x*t)
    #return np.exp(x/(t+3))*((1-x/(t+3)**2-1/(t+3)**2)*np.sin(x)-(2/(t+3))*np.cos(x))
    #return -u*x-3/2*t*t*u*u*u # - нелинейное
    #return np.exp(t) * np.sin(np.pi * x) - x * np.pi * np.exp(t)*np.cos(np.pi * x) + x*x/2 * np.exp(t) * np.pi * np.pi * np.sin(np.pi * x)  # Нужный пример
    #return -2*t
    #return np.sin(3*np.pi * x) * np.exp(-9*np.pi*np.pi * t)
    #return t*x+1-t**4/4
    return 0 # Локализация

def k(x, t, u = 0.0):
    #return u*u/2
    #return x*x/2   # Нужный пример
    #return t+0.5
    #return u+1
    return 0.5*u*u # Локализация 

def lim_matr(y):  
    min_el, max_el = y[0][0], y[0][0]

    for i in range(len(y)):
        for j in range(len(y[0])):
            if y[i][j] > max_el:
                max_el = y[i][j]
            if y[i][j] < min_el:
                min_el = y[i][j]
    
    return (min_el, max_el+1)

def plot(y, h, tao, N_h, N_t):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(xlim=(0, l), ylim=lim_matr(y))
    x = [i*h for i in range(N_h + 1)]

    for n_t in range(N_t + 1):
        ax.set_xlabel("x")
        #y_test = [np.sin(np.pi / l * test)*np.exp(-((np.pi / l)**2 *(n_t*tao))) for test in x]              
        #y_test = [np.sin(3*np.pi*test/2)+n_t*tao*test*test for test in x]
        #y_test = [np.cos(n_t*tao*test) for test in x]
        #y_test = [np.sin(test)*np.exp(test/(n_t*tao+3)) for test in x]
        #y_test = [np.exp(-n_t*tao*test) for test in x] #- нелинейное
        #y_test = [np.exp(n_t * tao) * np.sin(np.pi * test) for test in x] # Нужный пример, граничные условия - 4-е с конца
        #y_test = [n_t * tao * np.exp(-9 * np.pi * np.pi * tao * n_t) * np.sin(3 * np.pi * test) for test in x]
        #y_test = [test*test+n_t*tao-1 for test in x]
        #y_test = [n_t*n_t*tao*tao*test/2+n_t*tao for test in x]
        
        y_test = []            #Локализация
            
        for test in x:         #Локализация
            y_test.append(np.sqrt(4*(n_t*tao-test)) if test <= n_t*tao else 0.0)   #Локализация
    
        cur_graph_1, = ax.plot(x, y_test, color = 'green')
        cur_graph, = ax.plot(x, y[n_t], color='red')
        plt.draw()
        plt.pause(0.5)
        cur_graph.remove()
        cur_graph_1.remove()

def plot_steps(steps):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.set_xlabel("Слой")
    ax.set_ylabel("Шаг по времени")
    x = [i for i in range(1, len(steps)+1)]
    cur_graph, = ax.plot(x, steps, color ='blue')
    plt.show()

def check_currant(h, tao): # Проверка условия Куранта
    return True if tao / (h*h) <= 0.5 else False

def norma(y1, y2):
    m = abs(y1[0] - y2[0])
    for i in range(1, len(y1)):
        if (k := abs(y1[i] - y2[i])) > m:
            m = k

    return m

def get_norma(y_1, y_2):
    res = 0.0
    for i in range(len(y_1)):
        res += abs(y_1[i] - y_2[i])
    return res

def Yavnaya_sc(h, tao, N_h, N_t):
    y = []
    y.append([u_0(i*h) for i in range(N_h + 1)])

    for n_t in range(1, N_t + 1):
        cur_row = [phi_1(n_t*tao)] + [y[n_t-1][i]+tao/(h*h)*(y[n_t-1][i+1] - 2*y[n_t-1][i] + \
                    y[n_t-1][i-1]) + tao*f(i*h, (n_t-1)*tao) for i in range(1, N_h)] + [phi_2(n_t*tao)]
        y.append(cur_row)
    
    plot(y,h,tao,N_h,N_t)

    return y

def Neyavnaya_sc(h, tao, N_h, N_t):
    y = []
    y.append([u_0(i*h) for i in range(N_h + 1)])

    for n_t in range(1, N_t + 1):

        a, b, c = tao/(h*h), tao/(h*h), 1+2*tao/(h*h)
        alpha = [0.0]
        beta = [phi_1(n_t*tao)]

        for i in range(1, N_h + 1):
            if i == N_h:
                alpha.append(0.0)
                beta.append(phi_2(n_t*tao))
                break
            alpha.append(b/(c-a*alpha[i-1]))
            beta.append((y[n_t-1][i]+tao*f(i*h, (n_t-1)*tao)+a*beta[i-1])/(c-a*alpha[i-1]))
        cur_y = [beta[N_h]]

        for i in range(N_h-1, -1, -1):
            cur_y = [alpha[i]*cur_y[0]+beta[i]] + cur_y
        y.append(cur_y)
    plot(y,h,tao,N_h,N_t)
    return y

def Sc_with_weights(h, tao, N_h, N_t, sigma):
    y = []
    y.append([u_0(i*h) for i in range(N_h + 1)])

    for n_t in range(1, N_t + 1):
        a, b, c = tao*sigma/(h*h), tao*sigma/(h*h), 1+2*tao*sigma/(h*h)
        alpha = [0.0]
        beta = [phi_1(n_t*tao)]

        for i in range(1, N_h + 1):
            if i == N_h:
                alpha.append(0.0)
                beta.append(phi_2(n_t*tao))
                break
            alpha.append(b/(c-a*alpha[i-1]))
            beta.append((y[n_t-1][i]+tao*(1-sigma)/(h*h)*(y[n_t-1][i-1]-2*y[n_t-1][i]+y[n_t-1][i+1])+tao*f(i*h, (n_t-1/2)*tao)+a*beta[i-1])/(c-a*alpha[i-1]))
        cur_y = [beta[N_h]]

        for i in range(N_h-1, -1, -1):
            cur_y = [alpha[i]*cur_y[0]+beta[i]] + cur_y
        y.append(cur_y)
    
    plot(y,h,tao,N_h,N_t)
    return y

def Yavnaya_onestep_runge(y, h, tao, N_h, t, n_t):
    y_temp = y
    for j in range(n_t):
        y_temp = [phi_1(t+tao/n_t*(j+1))] + [y_temp[i]+tao/n_t/(h*h)*(y_temp[i+1] - 2*y_temp[i] + \
            y_temp[i-1]) + tao/n_t*f(i*h, t+tao*j/n_t) for i in range(1, N_h)] + [phi_2(t+tao/n_t*(j+1))]
    return y_temp

def Yavnaya_runge(y, h, tao, N_h, t, n_t, eps, N_t_max):
    y1 = Yavnaya_onestep_runge(y, h, tao, N_h, t, n_t)
    y2 = Yavnaya_onestep_runge(y, h, tao, N_h, t, n_t*2)
    n_t *= 2

    while (diff := norma(y1, y2)) >= eps and n_t*2 <= N_t_max:
        y1 = y2
        y2 = Yavnaya_onestep_runge(y, h, tao, N_h, t, n_t*2)
        n_t *= 2

    return y2, tao/n_t

def Neyavnaya_onestep_runge(y, h, tao, N_h, t, n_t):
    a, b, c = tao/n_t/(h*h), tao/n_t/(h*h), 1+2*tao/n_t/(h*h)
    cur_y = y
    for j in range(n_t):
        alpha = [0.0]
        beta = [phi_1(t + tao/n_t*(j+1))]

        for i in range(1, N_h + 1):
            if i == N_h:
                alpha.append(0.0)
                beta.append(phi_2(t + tao/n_t*(j+1)))
                break
            alpha.append(b/(c-a*alpha[i-1]))
            beta.append((cur_y[i]+tao/n_t*f(i*h, t+tao/n_t*j)+a*beta[i-1])/(c-a*alpha[i-1]))
        cur_y = [beta[N_h]]

        for i in range(N_h-1, -1, -1):
            cur_y = [alpha[i]*cur_y[0]+beta[i]] + cur_y
    
    return cur_y

def Neyavnaya_runge(y, h, tao, N_h, t, n_t, eps, N_t_max):
    y1 = Neyavnaya_onestep_runge(y, h, tao, N_h, t, n_t)
    y2 = Neyavnaya_onestep_runge(y, h, tao, N_h, t, n_t*2)
    n_t *= 2

    flag = True

    while (diff := norma(y1, y2)) >= eps and n_t*2 <= N_t_max:
        y1 = y2
        y2 = Neyavnaya_onestep_runge(y, h, tao, N_h, t, n_t*2)
        n_t *= 2
    return y2, tao/n_t

def Sc_with_weights_onestep_runge(y, h, tao, N_h, t, n_t, sigma):

    a, b, c = tao/n_t/(h*h)*sigma, tao/n_t/(h*h)*sigma, 1+2*tao/n_t/(h*h)*sigma
    cur_y = y
    for j in range(n_t):
        alpha = [0.0]
        beta = [phi_1(t + tao/n_t*(j+1))]

        for i in range(1, N_h + 1):
            if i == N_h:
                alpha.append(0.0)
                beta.append(phi_2(t + tao/n_t*(j+1)))
                break
            alpha.append(b/(c-a*alpha[i-1]))
            beta.append((cur_y[i]+tao*(1-sigma)/n_t/(h*h)*(y[i-1]-2*y[i]+y[i+1])+tao/n_t*f(i*h, t+tao/(2*n_t)*j)+a*beta[i-1])/(c-a*alpha[i-1]))
        cur_y = [beta[N_h]]

        for i in range(N_h-1, -1, -1):
            cur_y = [alpha[i]*cur_y[0]+beta[i]] + cur_y
    
    return cur_y

def Sc_with_weights_runge(y, h, tao, N_h, t, n_t, sigma, eps, N_t_max):
    y1 = Sc_with_weights_onestep_runge(y, h, tao, N_h, t, n_t, sigma)
    y2 = Sc_with_weights_onestep_runge(y, h, tao, N_h, t, n_t*2, sigma)
    n_t *= 2

    flag = True

    while (diff := norma(y1, y2)) >= eps and n_t*2 <= N_t_max:
        y1 = y2
        y2 = Sc_with_weights_onestep_runge(y, h, tao, N_h, t, n_t*2, sigma)
        n_t *= 2
    return y2, tao/n_t

def calc_coeff_aa(N_h, h, t, u):
    res = []
    for i in range(1, N_h+1):
        res.append((k(i*h, t, u[i]) + k((i-1)*h, t, u[i-1]))/2.0)
    return res
    
def convolution_a(h, N_h):
    a = []
    for i in range(1, N_h+1):
        a.append(Method_trap((i-1)*h, i*h))
    return a

def Balans(h, tao, N_h, N_t):
    a = convolution_a(h, N_h) 
    y = []
    y.append([u_0(i*h) for i in range(N_h + 1)])
    M = 3
    for n in range(1, N_t + 1):
        Coeff_for_a, Coeff_for_b, Coeff_for_c = [], [], []
        for i in range(N_h-1):
            Coeff_for_a.append(tao / (h*h) * a[i]((n-1) * tao, y[n-1][i]))
            Coeff_for_b.append(tao / (h*h) * a[i+1]((n-1) * tao, y[n-1][i+1]))
            Coeff_for_c.append(1+tao/(h*h)*(a[i]((n-1)*tao, y[n-1][i])+a[i+1]((n-1)*tao, y[n-1][i+1])))
        alpha = [0.0]
        beta = [phi_1(n*tao)]

        for i in range(1, N_h+1):
            if i == N_h:
                alpha.append(0.0)
                beta.append(phi_2(n*tao))
                break
            alpha.append(Coeff_for_b[i-1] / (Coeff_for_c[i-1] - Coeff_for_a[i-1]*alpha[i-1]))
            beta.append((y[n-1][i]+tao*f(i*h, (n-1) * tao, y[n-1][i]) + Coeff_for_a[i-1] * beta[i-1])/(Coeff_for_c[i-1] - Coeff_for_a[i-1] * alpha[i-1]))
        
        cur_y = [beta[N_h]]
        for i in range(N_h-1, -1, -1):
            cur_y = [alpha[i]*cur_y[0]+beta[i]] + cur_y

        y.append(cur_y)

    return y


def Method_trap(a, b):
    def int_t(t, u=0.0):
        return (k(a, t, u) + k(b, t, u))/2.0
    return int_t

def calc_a(x, h, t, u):
    return (k(x-h, t, u) + k(x, t, u)) / 2.0

def calc_coeff_a(h, N_h):
    a = []
    for i in range(1, N_h+1):
        a.append(Method_trap((i-1)*h, i*h))
    return a

def calc_1iter(h, tao, N_h, N_t, M, a, y, n):

    first_y = y

    for it in range(M):
        A, B, C, F = [], [0.0], [1.0], [phi_1(n*tao)]

        for i in range(1, N_h):
            if i == 1:
                pass

            A.append(tao/(h*h)*(k((i-1)*h, (n-1)*tao, y[i-1])+k(i*h, (n-1)*tao, y[i]))/2.0)
            B.append(tao/(h*h)*(k(i*h, (n-1)*tao, y[i])+k((i+1)*h, (n-1)*tao, y[i+1]))/2.0)
            C.append(1+tao/(h*h)*((k((i-1)*h, (n-1)*tao, y[i-1])+k(i*h, (n-1)*tao, y[i]))/2.0+(k(i*h, (n-1)*tao, y[i])+k((i+1)*h, (n-1)*tao, y[i+1]))/2.0))
            F.append(first_y[i] + tao*f(i*h, (n-1)*tao, y[i]))
        C.append(1.0)
        A.append(0.0)
        F.append(phi_2(n*tao))
 
        alpha = [0.0]
        beta = [phi_1(n*tao)]

        for i in range(1, N_h):
            alpha.append(B[i]/(C[i]-A[i-1]*alpha[i-1]))
            beta.append((F[i]+A[i-1]*beta[i-1])/(C[i]-A[i-1]*alpha[i-1]))
        
        beta.append(phi_2(n*tao))

        y = [beta[-1]]
        
        for i in range(N_h-1, -1, -1):

            y = [alpha[i]*y[0]+beta[i]] + y

    return y

def Balans_with_Newton(h, tao, N_h, N_t, M):
    a = calc_coeff_a(h, N_h) 

    M = M if M % 2 else M - 1
    y = []
    y.append([u_0(i*h) for i in range(N_h + 1)])

    for n in range(1, N_t + 1):

        y.append(calc_1iter(h, tao, N_h, N_t, M, a, y[n-1], n))

    return y


def Balans_Newton(h, tao, N_h, N_t, M):
    y, steps = [], []
    y.append([u_0(i*h) for i in range(N_h + 1)])

    for n in range(1, N_t + 1):
        k = 2

        y_1 = Newton_iter(h, tao, N_h, N_t, M, [], y[n-1], n, 1)
        y_2 = Newton_iter(h, tao, N_h, N_t, M, [], y[n-1], n, 2)

        while (s := get_norma(y_1, y_2)) > 1e-2 and k <= 1000:
            k *= 2

            y_1 = y_2
            y_2 = Newton_iter(h, tao, N_h, N_t, M, [], y[n-1], n, k)

        y.append(y_2)
        steps.append(tao/k)

    return y, steps


def Newton_iter(h, tao, N_h, N_t, M, a, y, n, z):
    
    step = 1.0 / z
    
    for time_iter in range(1, z+1):

        first_y = y

        for it in range(M):
            A, B, C, F = [], [0.0], [1.0], [phi_1((n-1+step*time_iter)*tao)]

            for i in range(1, N_h):
                
                if i == 1:
                    pass

                A.append(tao/(z*h*h)*(k((i-1)*h, (n-1+step*time_iter)*tao, y[i-1])+k(i*h, (n-1+step*time_iter)*tao, y[i]))/2.0)
                B.append(tao/(z*h*h)*(k(i*h, (n-1+step*time_iter)*tao, y[i])+k((i+1)*h, (n-1+step*time_iter)*tao, y[i+1]))/2.0)
                C.append(1+tao/(z*h*h)*((k((i-1)*h, (n-1+step*time_iter)*tao, y[i-1])+k(i*h, (n-1+step*time_iter)*tao, y[i]))/2.0+(k(i*h, (n-1+step*time_iter)*tao, y[i])+k((i+1)*h, (n-1+step*time_iter)*tao, y[i+1]))/2.0))
                F.append(first_y[i] + tao/z*f(i*h, (n-1+step*time_iter)*tao, y[i]))
            C.append(1.0)
            A.append(0.0)
            F.append(phi_2((n-1+step*time_iter)*tao))


            alpha = [0.0]
            beta = [phi_1((n-1+step*time_iter)*tao)]

            for i in range(1, N_h):
                alpha.append(B[i]/(C[i]-A[i-1]*alpha[i-1]))
                beta.append((F[i]+A[i-1]*beta[i-1])/(C[i]-A[i-1]*alpha[i-1]))
        
            beta.append(phi_2((n-1+step*time_iter)*tao))

            y = [beta[-1]]
        
            for i in range(N_h-1, -1, -1):

                y = [alpha[i]*y[0]+beta[i]] + y

    return y   



l = float(input("Введите длину отрезка: "))
T = float(input("Введите длину отрезка по времени: "))
N_h = int(input("Введите кол-во точек по пространству: "))
N_t = int(input("Введите кол-во точек по времени: "))

h = l / N_h
tao = T / N_t

print("Введите команду: ")
print("(Y): Явная схема")
print("(NY): Неявная схема")
print("(Weights): Схема с весами")
print("(Yc): Явная с контролем точности")
print("(NYc): Неявная с контролем точности")
print("(Balans): Интегро-интерполяционная схема")
print("(Balans_Newton): Нелинейная схема")

com = input("# ")
while True:
    if com == 'Y':
        h = l / N_h
        tao = T / N_t
        if not check_currant(h, tao):
            print('Не выполнено необходимое условие устойчивости 2*tao > h*h')
            N_h = int(input("Введите новое кол-во точек по пространству: "))
            N_t = int(input("Введите новое кол-во точек по времени: "))
            continue
        else:
            y = Yavnaya_sc(h, tao, N_h, N_t)
            break
    elif com == 'NY':
        y = Neyavnaya_sc(h, tao, N_h, N_t)
        break
    elif com == 'Weights':
        while True:
                sigma = float(input("Ввод параметра веса(сигма): "))
                if not 0 <= sigma <= 1:
                    print("Значение сигма должно принадлежать отрезку [0,1].")
                else:
                    break
        if sigma !=1 and (not check_currant(h, tao)):
            print('Не выполнено необходимое условие устойчивости 2*tao > h*h')
            N_h = int(input("Введите новое кол-во точек по пространству: "))
            N_t = int(input("Введите новое кол-во точек по времени: "))
            h = l / N_h
            tao = T / N_t
            continue
        y = Sc_with_weights(h, tao, N_h, N_t, sigma)
        break
    elif com == 'Yc':
        y = [[u_0(i*h) for i in range(N_h + 1)]]
        steps = []
        for n_t in range(1, N_t+1):
            cur_y, one_step = Yavnaya_runge(y[n_t-1], h, tao, N_h, (n_t-1)*tao, 1, 1e-5, 1e3)
            y.append(cur_y)
            steps.append(one_step)
        plot(y, h, tao, N_h, N_t)
        break
    elif com == 'NYc':
        y = [[u_0(i*h) for i in range(N_h + 1)]]
        steps = []
        for n_t in range(1, N_t+1):
            cur_y, one_step = Neyavnaya_runge(y[n_t-1], h, tao, N_h, (n_t-1)*tao, 1, 1e-4, 1e3)
            y.append(cur_y)
            steps.append(one_step)
        plot(y, h, tao, N_h, N_t)
    elif com == 'Balans':
        y = Balans(h, tao, N_h, N_t)
        plot(y, h, tao, N_h, N_t)
        break
    elif com == 'Balans_Newton':
        M = int(input("Кол-во итераций: "))
        y, steps = Balans_Newton(h, tao, N_h, N_t, M)
        plot(y, h, tao, N_h, N_t)
        break
    