## Simulation of sigle polarization optical signal transmission

-> instância de um objeto paramTX para adição de todos os parâmetros impostos
pelo transmissor, além da definição dos parâmetros do canal óptico:
	
	- Ordem do formato de modulação [M, int]
	- Taxa de símbolo [RS, int]
	- Amostras por símbolo [SpS, int]
	- Número total de bits por polarização [Nbits, int]
	- Filtro de modelagem de pulso [str]
	- Número de coeficientes de filtro de modelagem de pulso [int]
	- RRC rolloff [float]
	- Potência do sinal óptico [dBm]
	- Número de canais WDM [int]
	- Frequência central do espectro óptico [float]
	- Espaçamento da grade WDM

## Core simulation code

-> geração do sinal óptico e sua propagação no canal óptico, adição de ruído, detecção e demodulação do receptor.