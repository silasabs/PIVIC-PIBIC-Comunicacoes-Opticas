# Default_encoding: UTF-8
# Neural network functions

# Auxiliary functions

def generator():
  """
  Função que gera inteiros aleatórios.
  """
  return random.random()

# Activation functions

def heaviside(v_signal):
    """
    Função de ativação Heaviside 
    :param v_signal : campo local induzido do neurônio.
    
    :return activFunction: retorna valor 0 ou 1 dependendo do campo local induzido.
    """
    activFunction = np.heaviside(v_signal, 1)
    return activFunction

    # Forma alternativa
    # return 1 * (v_signal >= 0) - 1 * (v_signal < 0)

def sigmoid(v_signal, a):
  """
  Função de ativação sigmoid 
  :param v_signal : campo local induzido do neurônio.
  :param a : parâmetro de inclinação da função sigmoid.

  :return activFunction: retorna uma faixa contínua de valores 0 e 1.
  """
  return 1 / (1 + np.exp(-a * v_signal))


def signum(v_signal):
    """
    Função de ativação signum
    :param v_signal : campo local induzido do neurônio.

    :return activFunction: retorna valores na faixa de -1 e 1.
    """
    activFunction = np.sign(v_signal)
    return activFunction

# linear regression functions

def estimativa_coef(x, y):
  """
  Esta função usa o método dos mínimos quadrados como forma de estimar os coeficientes da reta de regressão
  param x, y: diagrama de dispersão.

  return b0, b1: coeficientes estimados.
  """
  # x e y recebem as  a variáveis de alvo e as variáveis independentes.
  # Contendo o diagrama de dispersão.
   .
  
  n = np.size(x) # número de observações visto que y depende de x

  # retirando a média dos valores de x e y
  mean_x = np.mean(x) 
  mean_y = np.mean(y)

  # calcula o desvio cruzado e o desvio sobre x
  SS_xy = np.sum(y * x) - n * mean_y * mean_x
  SS_xx = np.sum(x * x) - n * mean_x * mean_x

  # Determinação dos coeficientes
  b1 = SS_xy / SS_xx
  b0 = mean_y - b1 * mean_x

  return (b0, b1)


def plot_regression_line(x, y, b):
  """
  Esta função realiza o plot da reta de regressão sobre o diagrama de dispersão
  param x, y: diagrama de dispersão.
  param b: coeficientes estimados.
  """
  # Plot do diagrama de dispersão.
  plt.scatter(x, y)

  # Plot da reta de regressão.
  y = b[0] + b[1] * x
  plt.plot(x, y)

  plt.xlabel('x')
  plt.ylabel('y')

  plt.show()


def Gear_Sax(s, a, b, h):
    """
    Modified Gerchberg–Saxton (GS)

    param s: 
    param a: Intensidade do sinal recebido.
    param b: Medidas complementares após diferentes projeções
    param h: [ndarray] Função de transferência para dispersão cromática da fibra.

    return s: [ndarray]
    return x: [ndarray] Estimativa do sinal transmitido.
    """

    #shape = h.ndim

    b = np.array(b, dtype='complex').reshape(-1)
    a = np.array(a, dtype='complex').reshape(-1)

    bphase = np.angle(b)
    aphase = np.angle(a)

    try:
       h_inv = np.linalg.inv(h)
    except ValueError:
        print('matrix must be 2 dimensional square')

    # Compensação da disperção cromática.
    h_inv = np.array(h_inv).reshape(-1)
    h = np.array(h).reshape(-1)

    x = np.convolve(h_inv, s, mode='full')
    x = np.convolve(h, x, mode='full')
    # linha 04 ??
    s = np.convolve(h_inv, x, mode='full')
    d = np.convolve(h, s, mode='full')
    dphase = np.angle(d)

    # produto externo das fases de d(t)b(t)
    d = np.outer(dphase, bphase).reshape(-1)
    s = np.convolve(h_inv, d, mode='full')
    sphase = np.angle(s)
    # produto externo entre as fases de s(t)a(t)
    s = np.outer(sphase, aphase).reshape(-1)

    return s, x
