# PIVIC - Recuperação da fase de sinais ópticos  baseada em machine learning
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FPhase_retrieval&count_bg=%233B72CE&title_bg=%23000000&icon=wikipedia.svg&icon_color=%23FFFFFF&title=More+information&edge_flat=false)](https://hits.seeyoufarm.com) [![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

Neste projeto buscamos desenvolver e implementar métodos de PDS e aprendizado de máquina baseada em redes neurais artificiais (RNAs) para o problema de recuperação de fase de sinais ópticos.

## Algoritmo
A recuperação de fase é o processo de determinar, por meio de um algoritmo, a fase de um sinal quando apenas medidas de sua amplitude são conhecidas.  Dado um  sinal complexo ![equation](https://latex.codecogs.com/gif.latex?E%28t%29), de amplitude ![equation](https://latex.codecogs.com/gif.latex?%7CE%28t%29%7C) e fase ![equation](https://latex.codecogs.com/gif.latex?%5Cphi%28t%29), este pode ser representado por

![equation](https://latex.codecogs.com/gif.latex?E%28t%29%3D%7CE%28t%29%7Ce%5E%7B%5Cphi%28t%29%7D)

Assim se o valor absoluto ![equation](https://latex.codecogs.com/gif.latex?%7CE%28t%29%7C) for conhecido, ou medidas do valor absoluto de funções de ![equation](https://latex.codecogs.com/gif.latex?E%28t%29) forem conhecidas, um algoritmo de recuperação de fase pode ser utilizadado para conseguirmos determinar ![equation](https://latex.codecogs.com/gif.latex?%5Cphi%28t%29). Para que tudo isso seja possível geralmente requer-se que ![equation](https://latex.codecogs.com/gif.latex?E%28t%29) obedeça a certas condições, que dependerão de restrições impostas pelo problema.

Em sistemas de comunicações ópticas os algoritmos de recuperação de fase podem ser utilizados em conjunto com receptores DD para a implementação de detecção coerente, uma vez que tais algoritmos recuperam a fase do sinal recebido e, portanto, também a informação nela presente.

Aplicação de um algoritmo de recuperação de fase
[![Algoritmo.png](https://i.postimg.cc/L66XHDDB/Algoritmo.png)](https://postimg.cc/py4xCDgp)

Logo poderemos determinar se RNAs podem ser ferramentas úteis na construção de algoritmos de PDS para receptores ópticos de baixo custo e alta eficiência energética.

**Por favor, observar os [requisitos.](https://github.com/silasabs/PIVIC-Comunicacoes-Opticas/blob/main/requisitos.txt)**

