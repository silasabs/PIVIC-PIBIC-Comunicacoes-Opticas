# Resumo do Artigo
### Phase Retrieval Receiver Based on Deep Learning for Minimum-phase Signal Recovery
<br>
<center>European Conference on Optical Communication (ECOC) 2022 © </center>
</br>

**.BibTeX**

```
@report{, abstract = {We propose a deep learning-based phase retrieval receiver for minimum-phase signal recovery. Simulation results show that the HD-FEC limit at BER 3.8e-3 is achieved with 2-dB lower CSPR and 1.6-dB better receiver sensitivity compared to a conventional four-fold upsampled Kramers-Kronig receiver in relevant system settings.}, author = {Daniele Orsuti and Cristian Antonelli and Alessandro Chiuso and Marco Santagiustina and Antonio Mecozzi and Andrea Galtarossa and Luca Palmieri}, title = {Phase Retrieval Receiver Based on Deep Learning for Minimum-phase Signal Recovery}, }
```
<p align=justify><b>Objetivos:</b> Apresentar um esquema de recuperação de fase baseado em Deep Learning (DL) capaz de recostruir a fase de um sinal MP (fase mínima) com potência de portadora fraca (baixos valores de CSPR) além de obter melhorias para a sensibilidade do receptor NN (Neural Network) se comparado com o receptor KK (Kramers-Krnig) de quatro vezes upsampled convencional.</p>

## Introdução
<p align=justify>O artigo explora as fragilidades do método KK como o fator de upsampling digital relativamente alto sendo necessário para acomodar o alargamento espectral gerado pelas operações não lineares no algoritmo KK, nos testes realizados um fator de quatro upsampling  mostrou ser suficiente em casos práticos visto na referência <a href="https://ieeexplore.ieee.org/document/8346206">[03]</a>. Outro fato citado pelo autor e a necessidade de alta relação portadora potência de sinal (CSPR) para atender à condição MP, por tal restrição é introduzido uma penalidade de sensibilidade adicional aumentando o impacto dos efeitos de propagação de fibra não linear. <a href="https://ieeexplore.ieee.org/document/8274918">[04]</a> Tais perdas podem ser reduzidas modificando a cadeia DSP de recuperação de fase tal método sugere que melhorias podem ser obtidas afastando-se de
uma implementação teoricamente perfeita do receptor KK.</p>

## Método Proposto e Resumo da Solução

<center>Detalhes da CNN Temporal.</center>

<p align="center">
  <img src="https://i.postimg.cc/fWFH4zW8/Captura-de-tela-2022-10-16-105550.png">
</p>

<br>
<center>Parâmetros da Rede Neural Convolucional (CNN)</center>
</br>

- **D block:** downsampling.
- **U block:** upsampling.
- (Blocos D/U - Realizado pela aplicação repetida de camadas convolucionais não causais, **convolucionais transpostas** com passo 2. E selecionado 3 como o número de blocos de downsampling e upsampling. 72 amostras de entrada, cada camada possui kernel_size = 3 com 32 canais.)
- **Função a ser Minimizada:**  Erro quadrático médio.
- **Otimização:** Baseado em Adam 
- **Taxa de aprendizado:** $10^{-3}$
- **Tamanho de Lote:** 256 

    Para cada camada convolucional o número de canais e tamanho do kernel são 32 e 3, respectivamente.

## Parâmetros da Configuração
Valores fixados durante a simulação, CSPR: 2, 4, 10 *dB*


