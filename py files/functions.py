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
  Está função usa o método dos mínimos quadrados como forma de estimar os coeficientes da reta de regressão
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
  Está função realiza o plot da reta de regressão sobre o diagrama de dispersão
  param x, y: diagrama de dispersão.
  param b: coeficientes estimados.
  """
  # plot do diagrama de dispersão.
  plt.scatter(x, y)

  # plot da reta de regressão.
  y = b[0] + b[1] * x
  plt.plot(x, y)

  plt.xlabel('x')
  plt.ylabel('y')

  plt.show()