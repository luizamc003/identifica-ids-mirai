import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

class DataAnalyzer:
    def __init__(self, file_path, output_dir="output"):
        self.file_path = file_path
        self.output_dir = output_dir
        
        # Cria a pasta de resultados se não existir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Arquivo {self.file_path} não encontrado.")
            
        self.df = pd.read_csv(self.file_path)
        print(f"Análise iniciada para: {file_path}")

    def save_report(self):
        """Salva um resumo estatístico em arquivo de texto"""
        report_path = os.path.join(self.output_dir, "estatisticas_dataset.txt")
        with open(report_path, 'w') as f:
            f.write("=== RELATÓRIO DE ANÁLISE DO DATASET ===\n\n")
            f.write(f"Total de Registros: {self.df.shape[0]}\n")
            f.write(f"Colunas: {list(self.df.columns)}\n\n")
            f.write("--- Distribuição de Classes ---\n")
            f.write(self.df['label'].value_counts().to_string())
            f.write("\n\n--- Médias por Classe ---\n")
            f.write(self.df.groupby('label').mean().to_string())
        
        print(f"Relatório salvo em: {report_path}")

    def plot_class_distribution(self):
        plt.figure(figsize=(12, 6))
        order = self.df['label'].value_counts().index
        ax = sns.countplot(data=self.df, x='label', order=order, palette='magma')
        plt.title('Distribuição das Classes (Ataques Mirai vs Benigno)')
        plt.xticks(rotation=30)
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "distribuicao_classes.png"))
        plt.close() # Fecha para não sobrecarregar a memória

    def plot_correlation(self):
        plt.figure(figsize=(14, 10))
        # Seleciona apenas colunas numéricas para correlação
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', fmt=".2f", center=0)
        plt.title('Matriz de Correlação das Características de Rede')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "matriz_correlacao.png"))
        plt.close()

    def plot_feature_comparison(self, feature="Header_Length"):
        """Compara uma métrica específica entre tráfego Benigno e Ataques"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='label', y=feature, palette='Set2')
        plt.title(f'Comparação de {feature} por Classe')
        plt.xticks(rotation=30)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"comparativo_{feature}.png"))
        plt.close()

    def run_all(self):
        """Executa e salva todas as análises de uma vez"""
        self.save_report()
        self.plot_class_distribution()
        self.plot_correlation()
        self.plot_feature_comparison("Header_Length") # Fundamental para Mirai
        self.plot_feature_comparison("IAT")           # Fundamental para Mirai
        print(f"Todas as análises foram salvas na pasta '{self.output_dir}'")