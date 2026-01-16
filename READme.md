# IDS para Detecção de Ataques Mirai em Redes 5G/IoT com XGBoost

Este projeto implementa um Sistema de Detecção de Intrusão (IDS) focado na identificação de variantes do malware **Mirai** em ambientes de redes 5G e dispositivos IoT. Utilizando o algoritmo **XGBoost**, o sistema analisa fluxos de rede para classificar o tráfego entre legítimo e malicioso com alta precisão e baixo custo computacional.

## Contexto e Problema

A proliferação de dispositivos IoT com segurança fragilizada permitiu a ascensão da botnet **Mirai**, que sequestra estes aparelhos para lançar ataques de **DDoS** (Negação de Serviço Distribuída). Em redes 5G, a densidade de dispositivos torna esta ameaça ainda mais crítica, exigindo detecção em tempo real na camada de **Fog Computing**.

## Tecnologias Utilizadas

- **Linguagem:** Python 3.x
- **Modelo de ML:** XGBoost (Extreme Gradient Boosting)
- **Processamento de Dados:** Pandas, Scikit-learn
- **Dataset:** CICIoT2023 (focado em ataques IoT reais)
- **Ferramenta de Captura:** Pcap2csv (para consistência entre treino e inferência)

## Estrutura do Modelo

O modelo foi treinado utilizando uma seleção otimizada de **13 instâncias (features)** extraídas do tráfego de rede, conforme proposto no artigo base.

### Instâncias Chave:

- `rst_count`: Contagem de flags Reset (essencial para distinguir tráfego benigno).
- `Header_Length`: Comprimento do cabeçalho (identifica pacotes anómalos de flood).
- `Variance` / `Min`: Medem a repetitividade estatística do ataque.
- `UDP` / `TCP`: Identificadores de protocolo base.

### Classes Detectadas:

1. **BenignTraffic**: Tráfego legítimo.
2. **Mirai-greeth_flood**: Ataque via encapsulamento GRE (camada Ethernet).
3. **Mirai-greip_flood**: Ataque via encapsulamento GRE (camada IP).
4. **Mirai-udpplain**: Inundação direta via pacotes UDP.

## Performance e Resultados

O modelo foi submetido a um teste de stress (validação externa), apresentando os seguintes resultados:

- **Acurácia Geral:** 98.37%
- **Precisão Benigna:** 1.0000 (Zero Falsos Positivos)
- **Destaque:** O uso do parâmetro `tree_method='hist'` permitiu um treino extremamente rápido, tornando o modelo viável para hardware limitado.

## Referência

Este trabalho foi desenvolvido com base no artigo:

> _ZANCANELLA, M. et al. "Machine Learning for Mirai Attack Detection in 5G-Enabled IoT Networks". Wireless Days Conference, 2025._
