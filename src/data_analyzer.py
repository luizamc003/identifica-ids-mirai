import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
# tentar tirar os features correlacionadod
# 7usar log no boxsplot
class DataAnalyzer:
    def __init__(self, file_path, output_dir="output"):
        self.file_path = file_path
        self.output_dir = output_dir
        
        # Cria a pasta de resultados se não existir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Pasta '{self.output_dir}' criada")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Arquivo {self.file_path} não encontrado.")
        
        print(f"Carregando dataset: {file_path}")
        self.df = pd.read_csv(self.file_path)
        print(f"Análise iniciada: {len(self.df)} amostras, {len(self.df.columns)} colunas")
    
    def save_report(self):
        """Salva um resumo estatístico em arquivo de texto"""
        print("\nGerando relatório estatístico...")
        report_path = os.path.join(self.output_dir, "estatisticas_dataset.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DO DATASET - DETECÇÃO MIRAI\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total de Registros: {self.df.shape[0]:,}\n")
            f.write(f"Total de Colunas: {self.df.shape[1]}\n")
            f.write(f"\nColunas: {list(self.df.columns)}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("DISTRIBUIÇÃO DE CLASSES\n")
            f.write("-"*70 + "\n")
            class_counts = self.df['label'].value_counts()
            class_percentages = self.df['label'].value_counts(normalize=True) * 100
            
            for label in class_counts.index:
                count = class_counts[label]
                pct = class_percentages[label]
                f.write(f"{label:25s}: {count:8,} ({pct:5.2f}%)\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("ESTATÍSTICAS DESCRITIVAS POR CLASSE\n")
            f.write("-"*70 + "\n\n")
            f.write(self.df.groupby('label').describe().to_string())
            
            f.write("\n\n" + "-"*70 + "\n")
            f.write("MÉDIAS POR CLASSE\n")
            f.write("-"*70 + "\n\n")
            f.write(self.df.groupby('label').mean().to_string())
            
            f.write("\n\n" + "-"*70 + "\n")
            f.write("VALORES AUSENTES POR COLUNA\n")
            f.write("-"*70 + "\n\n")
            missing = self.df.isnull().sum()
            for col, count in missing.items():
                if count > 0:
                    f.write(f"{col:20s}: {count} ({count/len(self.df)*100:.2f}%)\n")
            if missing.sum() == 0:
                f.write("Nenhum valor ausente encontrado\n")
        
        print(f"Relatório salvo: {report_path}")
    
    def plot_class_distribution(self):
        """Gera gráfico de distribuição das classes"""
        print("\nGerando gráfico de distribuição de classes...")
        
        plt.figure(figsize=(14, 7))
        order = self.df['label'].value_counts().index
        ax = sns.countplot(data=self.df, x='label', order=order, palette='viridis')
        
        plt.title('Distribuição das Classes - Ataques Mirai vs Tráfego Benigno', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Classe', fontsize=12)
        plt.ylabel('Quantidade de Amostras', fontsize=12)
        plt.xticks(rotation=30, ha='right')
        
        # Adiciona valores sobre as barras
        for p in ax.patches:
            height = int(p.get_height())
            ax.annotate(f'{height:,}', 
                       (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='bottom', 
                       xytext=(0, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "distribuicao_classes.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo: {save_path}")
    
    def plot_correlation(self):
        """Gera matriz de correlação"""
        print("\nGerando matriz de correlação...")
        
        plt.figure(figsize=(16, 12))
        
        # Seleciona apenas colunas numéricas
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        # Máscara para triângulo superior
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Heatmap
        sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', 
                    fmt=".2f", center=0, square=True,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Matriz de Correlação das Características de Rede', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, "matriz_correlacao.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Matriz salva: {save_path}")
        
        # Identifica correlações altas
        print("\nCorrelações Altas (|r| > 0.95):")
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.95:
                    high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        
        if high_corr:
            for feat1, feat2, corr_val in high_corr:
                print(f"   {feat1:15s} x {feat2:15s}: {corr_val:.4f}")
        else:
            print("   Nenhuma correlação alta detectada")
    
    def plot_feature_comparison(self, feature="Header_Length", use_log=True):
        """
        Compara uma métrica específica entre tráfego Benigno e Ataques
        
        feature: nome da coluna a ser analisada
        use_log: se True, usa escala logarítmica no eixo Y
        """
        print(f"\nGerando boxplot: {feature} (log={use_log})...")
        
        if feature not in self.df.columns:
            print(f"Feature '{feature}' não encontrada no dataset")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Boxplot Normal
        sns.boxplot(data=self.df, x='label', y=feature, palette='Set2', ax=ax1)
        ax1.set_title(f'Comparação de {feature} por Classe\n(Escala Normal)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Classe', fontsize=10)
        ax1.set_ylabel(feature, fontsize=10)
        ax1.tick_params(axis='x', rotation=30)
        
        # Boxplot com Log
        if use_log:
            # Adiciona pequeno valor para evitar log(0)
            data_log = self.df.copy()
            min_positive = data_log[data_log[feature] > 0][feature].min()
            epsilon = min_positive / 10 if min_positive > 0 else 1e-10
            data_log[feature] = data_log[feature] + epsilon
            
            sns.boxplot(data=data_log, x='label', y=feature, palette='Set2', ax=ax2)
            ax2.set_yscale('log')
            ax2.set_title(f'Comparação de {feature} por Classe\n(Escala Logarítmica)', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel('Classe', fontsize=10)
            ax2.set_ylabel(f'{feature} (log)', fontsize=10)
            ax2.tick_params(axis='x', rotation=30)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f"comparativo_{feature}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Boxplot salvo: {save_path}")
        
        # Estatísticas por classe
        print(f"\n   Estatísticas de {feature} por classe:")
        for label in self.df['label'].unique():
            data_class = self.df[self.df['label'] == label][feature]
            print(f"   {label:25s}: média={data_class.mean():10.2f}, "
                  f"mediana={data_class.median():10.2f}, "
                  f"std={data_class.std():10.2f}")
    
    def plot_feature_distributions(self, features=None, use_log=True):
        """
        Plota distribuições de múltiplas features lado a lado
        """
        if features is None:
            # Usa features numéricas (exceto label se for numérica)
            features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'label' in features:
                features.remove('label')
        
        print(f"\nGerando distribuições para {len(features)} features...")
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            for label in self.df['label'].unique():
                data = self.df[self.df['label'] == label][feature]
                
                if use_log and data.min() > 0:
                    data = np.log10(data + 1)
                
                ax.hist(data, alpha=0.5, label=label, bins=30)
            
            ax.set_xlabel(f'{feature} {"(log10)" if use_log else ""}', fontsize=9)
            ax.set_ylabel('Frequência', fontsize=9)
            ax.set_title(feature, fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        # Remove eixos extras
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "distribuicoes_features.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distribuições salvas: {save_path}")
    
    def run_all(self, features_to_plot=None):
        """
        Executa e salva todas as análises de uma vez
        
        features_to_plot: lista de features para plotar boxplots detalhados
                         Se None, usa apenas Header_Length e Number
        """
        print("\n" + "="*70)
        print("INICIANDO ANALISE COMPLETA DO DATASET")
        print("="*70)
        
        # Análises básicas
        self.save_report()
        self.plot_class_distribution()
        self.plot_correlation()
        
        # Boxplots detalhados
        if features_to_plot is None:
            # Features importantes para detecção Mirai (sem Weight e IAT removidas)
            features_to_plot = ["Header_Length", "Number", "Duration", "Tot size"]
        
        for feature in features_to_plot:
            if feature in self.df.columns:
                self.plot_feature_comparison(feature, use_log=True)
        
        # Distribuições gerais
        self.plot_feature_distributions(use_log=True)
        
        print("\n" + "="*70)
        print(f"✅ ANÁLISE COMPLETA FINALIZADA!")
        print(f"📁 Todos os arquivos foram salvos em: '{self.output_dir}'")
        print("="*70)
