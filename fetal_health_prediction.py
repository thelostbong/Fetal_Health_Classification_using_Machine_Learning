#!/usr/bin/env python3
"""
Fetal Health Classification using Cardiotocography Features
===========================================================

Authors: M Ashish, Nishanth A
Dataset: Fetal Health Classification (Kaggle)
Technique: Multiple ML algorithms with hyperparameter tuning

This script provides a comprehensive analysis of fetal health classification
using cardiotocography (CTG) features with improved structure and performance.
"""

import os
import warnings
import time
import multiprocessing
from pathlib import Path

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                   cross_validate, GridSearchCV)
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_recall_fscore_support,
                           mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Imbalanced learning
import imblearn.pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Model persistence
import joblib

# Configuration
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Global variables for data storage
data = None
X_train = None
X_test = None
y_train = None
y_test = None
models = {}
results = {}
evaluation_results = {}
save_path = "fetal_health_results"

def setup_environment(output_path="fetal_health_results"):
    """Set up the environment and create output directory."""
    global save_path
    save_path = output_path
    
    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    print(f"Output directory created/verified: {save_path}")

def load_and_preprocess_data(data_path="fetal_health.csv"):
    """Load and preprocess the fetal health dataset."""
    global data
    print("Loading and preprocessing data...")
    start_time = time.time()
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"Dataset loaded with shape: {data.shape}")
    
    # Remove duplicates
    initial_size = len(data)
    data.drop_duplicates(keep='first', inplace=True)
    final_size = len(data)
    print(f"Removed {initial_size - final_size} duplicates. Final shape: {data.shape}")
    
    print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")

def explore_data():
    """Perform comprehensive exploratory data analysis."""
    global data
    print("\nPerforming exploratory data analysis...")
    
    # Basic statistics
    print("\nDataset Information:")
    print(f"Shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Data types:\n{data.dtypes.value_counts()}")
    
    # Target distribution
    target_dist = data['fetal_health'].value_counts().sort_index()
    print(f"\nTarget distribution:")
    for health_class, count in target_dist.items():
        health_label = {1.0: 'Normal', 2.0: 'Suspect', 3.0: 'Pathological'}[health_class]
        print(f"  {health_label} ({health_class}): {count} ({count/len(data)*100:.1f}%)")

def create_visualizations():
    """Create comprehensive visualizations of the dataset."""
    global data, save_path
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create a comprehensive visualization dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Target distribution
    ax1 = fig.add_subplot(gs[0, 0])
    target_counts = data['fetal_health'].value_counts().sort_index()
    colors = ['#2E8B57', '#FFD700', '#FF6347']
    bars = ax1.bar(['Normal', 'Suspect', 'Pathological'], target_counts.values, color=colors)
    ax1.set_title('Target Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Add value labels on bars
    for bar, value in zip(bars, target_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{value}\n({value/len(data)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Baseline heart rate distribution by class
    ax2 = fig.add_subplot(gs[0, 1])
    for i, health_class in enumerate([1.0, 2.0, 3.0]):
        data_subset = data[data['fetal_health'] == health_class]['baseline value']
        ax2.hist(data_subset, alpha=0.7, label=f'Class {int(health_class)}', 
                color=colors[i], bins=20)
    ax2.set_title('Baseline Heart Rate Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Baseline Heart Rate (bpm)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 3. Key features correlation heatmap
    ax3 = fig.add_subplot(gs[0, 2:])
    key_features = ['baseline value', 'accelerations', 'fetal_movement', 
                   'uterine_contractions', 'light_decelerations', 'severe_decelerations',
                   'abnormal_short_term_variability', 'mean_value_of_short_term_variability']
    corr_matrix = data[key_features + ['fetal_health']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
               fmt='.2f', ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Key Features Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 4. Feature distributions
    key_features_subset = ['accelerations', 'fetal_movement', 'uterine_contractions', 
                          'light_decelerations']
    for i, feature in enumerate(key_features_subset):
        ax = fig.add_subplot(gs[1, i])
        data.boxplot(column=feature, by='fetal_health', ax=ax)
        ax.set_title(f'{feature.replace("_", " ").title()}')
        ax.set_xlabel('Fetal Health Class')
        ax.set_ylabel(feature.replace("_", " ").title())
    
    # 5. Advanced features analysis
    histogram_features = ['histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency']
    for i, feature in enumerate(histogram_features):
        ax = fig.add_subplot(gs[2, i])
        for j, health_class in enumerate([1.0, 2.0, 3.0]):
            data_subset = data[data['fetal_health'] == health_class][feature]
            ax.hist(data_subset, alpha=0.6, label=f'Class {int(health_class)}', 
                   color=colors[j], bins=15)
        ax.set_title(f'{feature.replace("_", " ").title()}')
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # 6. Full correlation heatmap
    ax6 = fig.add_subplot(gs[3, :])
    # Select a subset of features for better visualization
    selected_features = data.columns[:-1]  # All features except target
    if len(selected_features) > 15:
        # Use correlation with target to select top features
        correlations = abs(data.corr()['fetal_health'].drop('fetal_health'))
        top_features = correlations.nlargest(15).index.tolist()
        selected_features = top_features
    
    full_corr = data[list(selected_features) + ['fetal_health']].corr()
    sns.heatmap(full_corr, annot=False, cmap='RdYlBu_r', center=0, 
               ax=ax6, cbar_kws={'shrink': 0.8})
    ax6.set_title('Top Features Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.suptitle('Fetal Health Classification - Comprehensive Data Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save the plot
    plt.savefig(os.path.join(save_path, 'comprehensive_data_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()

def prepare_data_for_modeling(test_size=0.2, random_state=42):
    """Prepare data for machine learning modeling."""
    global data, X_train, X_test, y_train, y_test
    print(f"\nPreparing data for modeling...")
    
    # Separate features and target
    X = data.drop('fetal_health', axis=1)
    y = data['fetal_health']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    # Show class distribution in splits
    train_dist = y_train.value_counts().sort_index()
    
    print("\nClass distribution in training set:")
    for cls, count in train_dist.items():
        print(f"  Class {int(cls)}: {count} ({count/len(y_train)*100:.1f}%)")

def train_models():
    """Train multiple machine learning models with hyperparameter tuning."""
    global X_train, y_train, models, results, save_path
    print("\nTraining machine learning models...")
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_jobs = min(multiprocessing.cpu_count() - 1, 4)  # Use up to 4 cores
    
    # Model configurations
    model_configs = {
        'logistic_regression': {
            'pipeline': ImbPipeline([
                ('scaler', StandardScaler()),
                ('sampler', SMOTE(random_state=42)),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'param_grid': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear']
            }
        },
        'random_forest': {
            'pipeline': ImbPipeline([
                ('scaler', StandardScaler()),
                ('sampler', SMOTE(random_state=42)),
                ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
            ]),
            'param_grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__class_weight': ['balanced']
            }
        },
        'svm': {
            'pipeline': ImbPipeline([
                ('scaler', StandardScaler()),
                ('sampler', SMOTE(random_state=42)),
                ('classifier', SVC(random_state=42, class_weight='balanced'))
            ]),
            'param_grid': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__gamma': ['scale', 'auto']
            }
        }
    }
    
    # Train each model
    for model_name, config in model_configs.items():
        print(f"\nTraining {model_name.replace('_', ' ').title()}...")
        start_time = time.time()
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            config['pipeline'],
            config['param_grid'],
            cv=cv,
            scoring=['accuracy', 'f1_weighted'],
            refit='f1_weighted',
            n_jobs=n_jobs,
            verbose=0
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Store results
        models[model_name] = grid_search.best_estimator_
        results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        # Save the model
        joblib.dump(grid_search.best_estimator_, 
                   os.path.join(save_path, f'{model_name}_model.joblib'))

def evaluate_models():
    """Evaluate all trained models on the test set."""
    global X_test, y_test, models, evaluation_results
    print("\nEvaluating models on test set...")
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name.replace('_', ' ').title()}:")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Store results
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report for {model_name.replace('_', ' ').title()}:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Normal', 'Suspect', 'Pathological']))

def create_evaluation_plots():
    """Create comprehensive evaluation visualizations."""
    global y_test, models, evaluation_results, save_path
    print("\nCreating evaluation plots...")
    
    # Set up the figure
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    model_names = list(models.keys())
    
    # Create confusion matrices
    for i, model_name in enumerate(model_names):
        y_pred = evaluation_results[model_name]['y_pred']
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Suspect', 'Pathological'],
                   yticklabels=['Normal', 'Suspect', 'Pathological'],
                   ax=axes[0, i])
        axes[0, i].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
        axes[0, i].set_xlabel('Predicted')
        axes[0, i].set_ylabel('Actual')
        
        # Model performance metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [evaluation_results[model_name][metric] for metric in metrics]
        
        bars = axes[1, i].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[1, i].set_title(f'{model_name.replace("_", " ").title()}\nPerformance Metrics')
        axes[1, i].set_ylabel('Score')
        axes[1, i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'model_evaluation.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Model comparison plot
    create_model_comparison_plot()

def create_model_comparison_plot():
    """Create a model comparison visualization."""
    global models, evaluation_results, save_path
    
    # Prepare data for comparison
    model_names = [name.replace('_', ' ').title() for name in models.keys()]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    comparison_data = []
    for model_name in models.keys():
        for metric in metrics:
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Metric': metric.replace('_', ' ').title(),
                'Score': evaluation_results[model_name][metric]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=comparison_df, x='Metric', y='Score', hue='Model')
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylim(0, 1)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.3f', rotation=0, padding=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'model_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report():
    """Generate a comprehensive summary report."""
    global models, evaluation_results, save_path, data
    print("\nGenerating comprehensive summary report...")
    
    report_lines = []
    report_lines.append("FETAL HEALTH CLASSIFICATION - ANALYSIS SUMMARY")
    report_lines.append("=" * 60)
    
    # Dataset information
    report_lines.append(f"\nDATASET INFORMATION:")
    report_lines.append(f"Dataset shape: {data.shape}")
    report_lines.append(f"Number of features: {data.shape[1] - 1}")
    report_lines.append(f"Number of samples: {data.shape[0]}")
    
    # Target distribution
    target_dist = data['fetal_health'].value_counts().sort_index()
    report_lines.append(f"\nTARGET DISTRIBUTION:")
    for health_class, count in target_dist.items():
        health_label = {1.0: 'Normal', 2.0: 'Suspect', 3.0: 'Pathological'}[health_class]
        percentage = count/len(data)*100
        report_lines.append(f"  {health_label}: {count} samples ({percentage:.1f}%)")
    
    # Model performance
    if evaluation_results:
        report_lines.append(f"\nMODEL PERFORMANCE COMPARISON:")
        report_lines.append(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        report_lines.append("-" * 70)
        
        for model_name, results in evaluation_results.items():
            report_lines.append(
                f"{model_name.replace('_', ' ').title():<20} "
                f"{results['accuracy']:<10.4f} "
                f"{results['precision']:<10.4f} "
                f"{results['recall']:<10.4f} "
                f"{results['f1_score']:<10.4f}"
            )
        
        # Best model
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['f1_score'])
        report_lines.append(f"\nBEST PERFORMING MODEL: {best_model[0].replace('_', ' ').title()}")
        report_lines.append(f"F1-Score: {best_model[1]['f1_score']:.4f}")
    
    # Feature importance (if Random Forest is available)
    if 'random_forest' in models:
        try:
            # Handle both Pipeline and direct model objects
            rf_model = models['random_forest']
            if hasattr(rf_model, 'named_steps'):
                # It's a Pipeline, get the classifier step
                feature_importance = rf_model.named_steps['classifier'].feature_importances_
            else:
                # It's a direct model
                feature_importance = rf_model.feature_importances_
                
            feature_names = X_train.columns
            importance_pairs = list(zip(feature_names, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            report_lines.append(f"\nTOP 10 MOST IMPORTANT FEATURES (Random Forest):")
            for i, (feature, importance) in enumerate(importance_pairs[:10]):
                report_lines.append(f"  {i+1:2d}. {feature:<35} {importance:.4f}")
        except Exception as e:
            report_lines.append(f"\nFeature importance could not be extracted: {e}")
    
    # Save report
    report_content = "\n".join(report_lines)
    with open(os.path.join(save_path, 'analysis_report.txt'), 'w') as f:
        f.write(report_content)
    
    print("Summary report saved to analysis_report.txt")
    print("\nKey findings:")
    if evaluation_results:
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['f1_score'])
        print(f"- Best model: {best_model[0].replace('_', ' ').title()} (F1: {best_model[1]['f1_score']:.4f})")
    print(f"- Dataset contains {data.shape[0]} samples with {data.shape[1]-1} features")
    print(f"- Class distribution: Normal: {target_dist[1.0]} ({target_dist[1.0]/len(data)*100:.1f}%)")

def create_comprehensive_pdf_report():
    """Create a comprehensive PDF report with all visualizations."""
    global data, save_path, X_train, y_train, X_test, y_test, models, evaluation_results
    
    print("\nCreating comprehensive PDF report with all visualizations...")
    pdf_filename = os.path.join(save_path, 'fetal_health_analysis_report.pdf')
    
    with PdfPages(pdf_filename) as pdf:
        # Set common style parameters
        plt.style.use('seaborn-v0_8')
        colors = ['#2E8B57', '#FFD700', '#FF6347']  # Normal, Suspect, Pathological
        class_names = ['Normal', 'Suspect', 'Pathological']
        class_values = [1.0, 2.0, 3.0]
        
        # Page 1: Target Class Distribution
        print("Creating Page 1: Target Class Distribution...")
        fig, ax = plt.subplots(figsize=(10, 8))
        target_counts = data['fetal_health'].value_counts().sort_index()
        bars = ax.bar(class_names, target_counts.values, color=colors)
        ax.set_title('Fetal Health - Target Class Distribution', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_xlabel('Fetal Health Status', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, target_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value}\n({value/len(data)*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Baseline Heart Rate Distribution
        print("Creating Page 2: Baseline Heart Rate Distribution...")
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, health_class in enumerate(class_values):
            data_subset = data[data['fetal_health'] == health_class]['baseline value']
            ax.hist(data_subset, alpha=0.7, label=f'{class_names[i]}', 
                    color=colors[i], bins=25, density=True)
        ax.set_title('Baseline Heart Rate Distribution by Fetal Health Class', fontsize=16, fontweight='bold')
        ax.set_xlabel('Baseline Heart Rate (bpm)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Key Features Correlation Heatmap
        print("Creating Page 3: Key Features Correlation...")
        fig, ax = plt.subplots(figsize=(12, 10))
        key_features = ['baseline value', 'accelerations', 'fetal_movement', 
                       'uterine_contractions', 'light_decelerations', 'severe_decelerations',
                       'abnormal_short_term_variability', 'mean_value_of_short_term_variability']
        corr_matrix = data[key_features + ['fetal_health']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   fmt='.2f', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Key CTG Features Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4: Feature Distribution Box Plots
        print("Creating Page 4: Feature Distribution Box Plots...")
        key_features_subset = ['accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(key_features_subset):
            data.boxplot(column=feature, by='fetal_health', ax=axes[i])
            axes[i].set_title(f'{feature.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Fetal Health Class', fontsize=12)
            axes[i].set_ylabel(feature.replace("_", " ").title(), fontsize=12)
        
        plt.suptitle('CTG Feature Distributions by Fetal Health Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 5: Histogram Features Analysis
        print("Creating Page 5: Histogram Features Analysis...")
        histogram_features = ['histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(histogram_features):
            for j, health_class in enumerate(class_values):
                data_subset = data[data['fetal_health'] == health_class][feature]
                axes[i].hist(data_subset, alpha=0.6, label=f'{class_names[j]}', 
                           color=colors[j], bins=20, density=True)
            axes[i].set_title(f'{feature.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(feature.replace("_", " ").title(), fontsize=12)
            axes[i].set_ylabel('Density', fontsize=12)
            if i == 0:
                axes[i].legend()
        
        plt.suptitle('CTG Histogram Features Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 6: Complete Features Correlation Heatmap
        print("Creating Page 6: Complete Features Correlation...")
        fig, ax = plt.subplots(figsize=(14, 12))
        selected_features = data.columns[:-1]  # All features except target
        if len(selected_features) > 15:
            # Use correlation with target to select top features
            correlations = abs(data.corr()['fetal_health'].drop('fetal_health'))
            top_features = correlations.nlargest(15).index.tolist()
            selected_features = top_features
        
        full_corr = data[list(selected_features) + ['fetal_health']].corr()
        sns.heatmap(full_corr, annot=False, cmap='RdYlBu_r', center=0, 
                   ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Top CTG Features Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 7: SMOTE Before/After Analysis - Basic Overview (if data is available)
        if X_train is not None and y_train is not None:
            print("Creating Page 7: SMOTE Before/After Analysis - Overview...")
            # Create SMOTE visualization
            from imblearn.over_sampling import SMOTE
            from sklearn.preprocessing import StandardScaler
            
            smote = SMOTE(random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Before SMOTE
            before_counts = y_train.value_counts().sort_index()
            bars1 = axes[0, 0].bar(class_names, before_counts.values, color=colors)
            axes[0, 0].set_title('Class Distribution BEFORE SMOTE', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('Count')
            for bar, value in zip(bars1, before_counts.values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               f'{value}\n({value/len(y_train)*100:.1f}%)', 
                               ha='center', va='bottom', fontweight='bold')
            
            # After SMOTE
            after_counts = pd.Series(y_train_smote).value_counts().sort_index()
            bars2 = axes[0, 1].bar(class_names, after_counts.values, color=colors)
            axes[0, 1].set_title('Class Distribution AFTER SMOTE', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Count')
            for bar, value in zip(bars2, after_counts.values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               f'{value}\n({value/len(y_train_smote)*100:.1f}%)', 
                               ha='center', va='bottom', fontweight='bold')
            
            # Sample size comparison
            comparison_data = [len(y_train), len(y_train_smote)]
            bars3 = axes[1, 0].bar(['Before SMOTE', 'After SMOTE'], comparison_data, 
                                  color=['lightcoral', 'lightblue'])
            axes[1, 0].set_title('Total Sample Size Comparison', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Total Samples')
            for bar, value in zip(bars3, comparison_data):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                               f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # Balance ratio
            before_ratios = [before_counts[cls]/before_counts.max() for cls in class_values]
            after_ratios = [after_counts[cls]/after_counts.max() for cls in class_values]
            x = np.arange(len(class_names))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, before_ratios, width, label='Before SMOTE', 
                          color='lightcoral', alpha=0.7)
            axes[1, 1].bar(x + width/2, after_ratios, width, label='After SMOTE', 
                          color='lightblue', alpha=0.7)
            axes[1, 1].set_title('Class Balance Ratio', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Ratio to Majority Class')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(class_names)
            axes[1, 1].legend()
            
            # Add ratio labels
            for bars in [axes[1, 1].containers[0], axes[1, 1].containers[1]]:
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, height + 0.02,
                                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
            
            plt.suptitle('SMOTE Impact Analysis - Overview', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 8: SMOTE Feature Distribution Analysis
            print("Creating Page 8: SMOTE Feature Distribution Analysis...")
            key_features = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions']
            feature_indices = [X_train.columns.get_loc(feat) for feat in key_features if feat in X_train.columns]
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for i, (feature_name, feature_idx) in enumerate(zip(key_features[:4], feature_indices[:4])):
                if i < 4:  # Ensure we don't exceed subplot count
                    # Before SMOTE distributions
                    for j, (class_val, class_name, color) in enumerate(zip(class_values, class_names, colors)):
                        class_data_before = X_train_scaled[y_train == class_val, feature_idx]
                        axes[i].hist(class_data_before, alpha=0.4, label=f'{class_name} (Before)', 
                                   color=color, bins=20, density=True, histtype='step', linewidth=2)
                        
                        # After SMOTE distributions  
                        class_data_after = X_train_smote[y_train_smote == class_val, feature_idx]
                        axes[i].hist(class_data_after, alpha=0.6, label=f'{class_name} (After)', 
                                   color=color, bins=20, density=True, histtype='stepfilled')
                    
                    axes[i].set_title(f'{feature_name.replace("_", " ").title()} Distribution', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel('Normalized Value')
                    axes[i].set_ylabel('Density')
                    if i == 0:  # Only show legend on first subplot to avoid clutter
                        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.suptitle('SMOTE Impact on Feature Distributions', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 9: SMOTE 2D Feature Relationships
            print("Creating Page 9: SMOTE 2D Feature Relationships...")
            scatter_features = [
                ('baseline value', 'accelerations'),
                ('fetal_movement', 'uterine_contractions'),
                ('light_decelerations', 'severe_decelerations'),
                ('abnormal_short_term_variability', 'mean_value_of_short_term_variability')
            ]
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for i, (feat1, feat2) in enumerate(scatter_features):
                if i < 4 and feat1 in X_train.columns and feat2 in X_train.columns:
                    feat1_idx = X_train.columns.get_loc(feat1)
                    feat2_idx = X_train.columns.get_loc(feat2)
                    
                    # Plot before SMOTE (smaller points, less alpha)
                    for j, (class_val, class_name, color) in enumerate(zip(class_values, class_names, colors)):
                        before_mask = y_train == class_val
                        axes[i].scatter(X_train_scaled[before_mask, feat1_idx], 
                                      X_train_scaled[before_mask, feat2_idx],
                                      c=color, alpha=0.3, s=10, label=f'{class_name} (Before)', 
                                      marker='o', edgecolors='none')
                        
                        # Plot after SMOTE (larger points, more alpha)
                        after_mask = y_train_smote == class_val
                        axes[i].scatter(X_train_smote[after_mask, feat1_idx], 
                                      X_train_smote[after_mask, feat2_idx],
                                      c=color, alpha=0.7, s=15, label=f'{class_name} (After)', 
                                      marker='s', edgecolors='black', linewidths=0.5)
                    
                    axes[i].set_title(f'{feat1.replace("_", " ").title()} vs {feat2.replace("_", " ").title()}', 
                                     fontsize=12, fontweight='bold')
                    axes[i].set_xlabel(feat1.replace("_", " ").title())
                    axes[i].set_ylabel(feat2.replace("_", " ").title())
                    if i == 0:  # Only show legend on first subplot
                        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.suptitle('SMOTE Impact on Feature Relationships', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Page 10: Model Performance Evaluation (if models are available)
        if models and evaluation_results:
            print("Creating Page 10: Model Performance Evaluation...")
            n_models = len(models)
            fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
            
            if n_models == 1:
                axes = axes.reshape(2, 1)
            
            model_names = list(models.keys())
            
            for i, model_name in enumerate(model_names):
                y_pred = evaluation_results[model_name]['y_pred']
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names,
                           yticklabels=class_names,
                           ax=axes[0, i])
                axes[0, i].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix',
                                    fontsize=14, fontweight='bold')
                axes[0, i].set_xlabel('Predicted')
                axes[0, i].set_ylabel('Actual')
                
                # Performance metrics
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                values = [evaluation_results[model_name][metric] for metric in metrics]
                metric_colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
                
                bars = axes[1, i].bar(metrics, values, color=metric_colors)
                axes[1, i].set_title(f'{model_name.replace("_", " ").title()}\nPerformance Metrics',
                                    fontsize=14, fontweight='bold')
                axes[1, i].set_ylabel('Score')
                axes[1, i].set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[1, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle('Machine Learning Model Performance Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 11: Model Comparison
            print("Creating Page 11: Model Performance Comparison...")
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data for comparison
            comparison_data = []
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for model_name in evaluation_results.keys():
                for metric in metrics:
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': evaluation_results[model_name][metric]
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            sns.barplot(data=comparison_df, x='Metric', y='Score', hue='Model', ax=ax)
            ax.set_title('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xlabel('Performance Metrics', fontsize=12)
            ax.set_ylim(0, 1)
            ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', rotation=0, padding=3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 12: Feature Importance (if Random Forest is available)
            if 'random_forest' in models:
                print("Creating Page 12: Feature Importance Analysis...")
                try:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Handle both Pipeline and direct model objects
                    rf_model = models['random_forest']
                    if hasattr(rf_model, 'named_steps'):
                        # It's a Pipeline, get the classifier step
                        feature_importance = rf_model.named_steps['classifier'].feature_importances_
                    else:
                        # It's a direct model
                        feature_importance = rf_model.feature_importances_
                        
                    feature_names = X_train.columns
                    importance_pairs = list(zip(feature_names, feature_importance))
                    importance_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top 15 features
                    top_features = importance_pairs[:15]
                    features, importances = zip(*top_features)
                    
                    bars = ax.barh(range(len(features)), importances)
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
                    ax.set_xlabel('Feature Importance', fontsize=12)
                    ax.set_title('Top 15 Most Important CTG Features (Random Forest)', fontsize=16, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for i, (bar, imp) in enumerate(zip(bars, importances)):
                        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                               f'{imp:.3f}', ha='left', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not create feature importance plot: {e}")
                    # Create a placeholder page
                    fig, ax = plt.subplots(figsize=(12, 10))
                    ax.text(0.5, 0.5, f'Feature Importance Analysis\nUnavailable\n\nError: {e}', 
                           ha='center', va='center', fontsize=14,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
                    ax.set_title('Feature Importance Analysis', fontsize=16, fontweight='bold')
                    ax.axis('off')
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        # Page 13: Summary Statistics
        print("Creating Page 13: Summary Statistics...")
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        # Create summary text
        summary_text = []
        summary_text.append("FETAL HEALTH CLASSIFICATION - COMPREHENSIVE ANALYSIS SUMMARY")
        summary_text.append("=" * 80)
        summary_text.append("")
        summary_text.append("DATASET OVERVIEW:")
        summary_text.append(f"• Total samples: {data.shape[0]}")
        summary_text.append(f"• Number of features: {data.shape[1] - 1}")
        summary_text.append(f"• Missing values: {data.isnull().sum().sum()}")
        summary_text.append("")
        
        # Target distribution
        target_dist = data['fetal_health'].value_counts().sort_index()
        summary_text.append("CLASS DISTRIBUTION:")
        for health_class, count in target_dist.items():
            health_label = {1.0: 'Normal', 2.0: 'Suspect', 3.0: 'Pathological'}[health_class]
            percentage = count/len(data)*100
            summary_text.append(f"• {health_label}: {count} samples ({percentage:.1f}%)")
        summary_text.append("")
        
        # Model performance (if available)
        if evaluation_results:
            summary_text.append("MODEL PERFORMANCE SUMMARY:")
            for model_name, results in evaluation_results.items():
                summary_text.append(f"• {model_name.replace('_', ' ').title()}:")
                summary_text.append(f"  - Accuracy: {results['accuracy']:.4f}")
                summary_text.append(f"  - Precision: {results['precision']:.4f}")
                summary_text.append(f"  - Recall: {results['recall']:.4f}")
                summary_text.append(f"  - F1-Score: {results['f1_score']:.4f}")
                summary_text.append("")
            
            # Best model
            best_model = max(evaluation_results.items(), key=lambda x: x[1]['f1_score'])
            summary_text.append("BEST PERFORMING MODEL:")
            summary_text.append(f"• {best_model[0].replace('_', ' ').title()}")
            summary_text.append(f"• F1-Score: {best_model[1]['f1_score']:.4f}")
            summary_text.append("")
        
        # Key insights
        summary_text.append("KEY INSIGHTS:")
        summary_text.append("• CTG features show significant correlation with fetal health status")
        summary_text.append("• Baseline heart rate, accelerations, and variability are key indicators")
        summary_text.append("• SMOTE technique effectively addresses class imbalance")
        summary_text.append("• Machine learning models achieve high accuracy for fetal health classification")
        summary_text.append("• Random Forest typically provides the best balance of performance metrics")
        
        # Display summary
        summary_str = "\n".join(summary_text)
        ax.text(0.05, 0.95, summary_str, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"Comprehensive PDF report created successfully!")
    print(f"Report saved as: {pdf_filename}")
    
    # Calculate total pages based on content
    base_pages = 6  # Pages 1-6: Basic data analysis
    smote_pages = 3 if X_train is not None and y_train is not None else 0  # Pages 7-9: SMOTE analysis
    model_pages = 3 if models and evaluation_results else 0  # Pages 10-12: Model analysis
    summary_pages = 1  # Page 13: Summary
    total_pages = base_pages + smote_pages + model_pages + summary_pages
    
    print(f"Total pages: {total_pages}")
    
    return pdf_filename

def generate_pdf_from_existing_results(data_path="fetal_health.csv", output_path="fetal_health_results"):
    """Generate PDF report from existing analysis results without re-running the full analysis."""
    global data, save_path, X_train, y_train, X_test, y_test, models, evaluation_results
    
    print("\nGenerating PDF report from existing results...")
    
    # Set up environment
    save_path = output_path
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        print(f"Dataset loaded: {data.shape}")
        
        # Try to load existing models and results
        try:
            # Load models if they exist
            model_files = {
                'logistic_regression': 'logistic_regression_model.joblib',
                'svm': 'svm_model.joblib', 
                'random_forest': 'random_forest_model.joblib'
            }
            
            models = {}
            for model_name, filename in model_files.items():
                model_path = os.path.join(output_path, filename)
                if os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} model")
            
            # Create dummy evaluation results if models exist but results don't
            if models:
                print("Creating PDF with available models...")
                # Note: For PDF generation, we'll skip model evaluation pages if test data isn't available
                evaluation_results = {}
                X_train, X_test, y_train, y_test = None, None, None, None
                
        except Exception as e:
            print(f"Could not load existing models: {e}")
            models = {}
            evaluation_results = {}
            
        # Generate the PDF report
        pdf_filename = create_comprehensive_pdf_report()
        return pdf_filename
        
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return None

def visualize_before_after_smote():
    """Create visualizations comparing data distribution before and after SMOTE."""
    global X_train, y_train, save_path
    print("\nCreating before/after SMOTE visualizations...")
    
    # Create SMOTE instance for demonstration
    smote = SMOTE(random_state=42)
    scaler = StandardScaler()
    
    # Scale the original data first
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply SMOTE
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # Create comprehensive before/after visualization
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Colors for classes
    colors = ['#2E8B57', '#FFD700', '#FF6347']  # Normal, Suspect, Pathological
    class_names = ['Normal', 'Suspect', 'Pathological']
    class_values = [1.0, 2.0, 3.0]
    
    # 1. Class distribution before SMOTE
    ax1 = fig.add_subplot(gs[0, 0])
    before_counts = y_train.value_counts().sort_index()
    bars1 = ax1.bar(class_names, before_counts.values, color=colors)
    ax1.set_title('Class Distribution BEFORE SMOTE', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Fetal Health Class')
    
    # Add value labels on bars
    for bar, value in zip(bars1, before_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value}\n({value/len(y_train)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Class distribution after SMOTE
    ax2 = fig.add_subplot(gs[0, 1])
    after_counts = pd.Series(y_train_smote).value_counts().sort_index()
    bars2 = ax2.bar(class_names, after_counts.values, color=colors)
    ax2.set_title('Class Distribution AFTER SMOTE', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Fetal Health Class')
    
    # Add value labels on bars
    for bar, value in zip(bars2, after_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value}\n({value/len(y_train_smote)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Sample size comparison
    ax3 = fig.add_subplot(gs[0, 2])
    comparison_data = [len(y_train), len(y_train_smote)]
    bars3 = ax3.bar(['Before SMOTE', 'After SMOTE'], comparison_data, 
                    color=['lightcoral', 'lightblue'])
    ax3.set_title('Total Sample Size Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Total Samples')
    
    # Add value labels
    for bar, value in zip(bars3, comparison_data):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Imbalance ratio visualization
    ax4 = fig.add_subplot(gs[0, 3])
    before_ratios = [before_counts[cls]/before_counts.max() for cls in class_values]
    after_ratios = [after_counts[cls]/after_counts.max() for cls in class_values]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    bars4a = ax4.bar(x - width/2, before_ratios, width, label='Before SMOTE', 
                     color='lightcoral', alpha=0.7)
    bars4b = ax4.bar(x + width/2, after_ratios, width, label='After SMOTE', 
                     color='lightblue', alpha=0.7)
    
    ax4.set_title('Class Balance Ratio', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Ratio to Majority Class')
    ax4.set_xlabel('Fetal Health Class')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    # Add ratio labels
    for bars in [bars4a, bars4b]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 5-8. Feature distribution comparisons for key features
    key_features = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions']
    feature_indices = [X_train.columns.get_loc(feat) for feat in key_features if feat in X_train.columns]
    
    for i, (feature_name, feature_idx) in enumerate(zip(key_features[:4], feature_indices[:4])):
        if i < 4:  # Ensure we don't exceed subplot count
            ax = fig.add_subplot(gs[1, i])
            
            # Before SMOTE distributions
            for j, (class_val, class_name, color) in enumerate(zip(class_values, class_names, colors)):
                class_data_before = X_train_scaled[y_train == class_val, feature_idx]
                ax.hist(class_data_before, alpha=0.4, label=f'{class_name} (Before)', 
                       color=color, bins=20, density=True, histtype='step', linewidth=2)
                
                # After SMOTE distributions  
                class_data_after = X_train_smote[y_train_smote == class_val, feature_idx]
                ax.hist(class_data_after, alpha=0.6, label=f'{class_name} (After)', 
                       color=color, bins=20, density=True, histtype='stepfilled')
            
            ax.set_title(f'{feature_name.title()} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Normalized Value')
            ax.set_ylabel('Density')
            if i == 0:  # Only show legend on first subplot to avoid clutter
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 9-12. 2D scatter plots for feature relationships
    scatter_features = [
        ('baseline value', 'accelerations'),
        ('fetal_movement', 'uterine_contractions'),
        ('light_decelerations', 'severe_decelerations'),
        ('abnormal_short_term_variability', 'mean_value_of_short_term_variability')
    ]
    
    for i, (feat1, feat2) in enumerate(scatter_features):
        if i < 4 and feat1 in X_train.columns and feat2 in X_train.columns:
            ax = fig.add_subplot(gs[2, i])
            
            feat1_idx = X_train.columns.get_loc(feat1)
            feat2_idx = X_train.columns.get_loc(feat2)
            
            # Plot before SMOTE (smaller points, less alpha)
            for j, (class_val, class_name, color) in enumerate(zip(class_values, class_names, colors)):
                before_mask = y_train == class_val
                ax.scatter(X_train_scaled[before_mask, feat1_idx], 
                          X_train_scaled[before_mask, feat2_idx],
                          c=color, alpha=0.3, s=10, label=f'{class_name} (Before)', 
                          marker='o', edgecolors='none')
                
                # Plot after SMOTE (larger points, more alpha)
                after_mask = y_train_smote == class_val
                ax.scatter(X_train_smote[after_mask, feat1_idx], 
                          X_train_smote[after_mask, feat2_idx],
                          c=color, alpha=0.7, s=15, label=f'{class_name} (After)', 
                          marker='s', edgecolors='black', linewidths=0.5)
            
            ax.set_title(f'{feat1.title()} vs {feat2.title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel(feat1.title())
            ax.set_ylabel(feat2.title())
            if i == 0:  # Only show legend on first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle('Before vs After SMOTE: Comprehensive Data Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save the plot
    plt.savefig(os.path.join(save_path, 'before_after_smote_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nSMOTE Application Summary:")
    print(f"{'='*50}")
    print(f"Original training samples: {len(y_train)}")
    print(f"After SMOTE samples: {len(y_train_smote)}")
    print(f"Increase: {len(y_train_smote) - len(y_train)} samples ({((len(y_train_smote) - len(y_train))/len(y_train)*100):.1f}%)")
    
    print(f"\nClass distribution changes:")
    for class_val, class_name in zip(class_values, class_names):
        before_count = sum(y_train == class_val)
        after_count = sum(y_train_smote == class_val)
        print(f"{class_name:>12}: {before_count:>4} → {after_count:>4} (+{after_count-before_count:>3})")

def run_complete_analysis(data_path="fetal_health.csv", output_path="fetal_health_results"):
    """Run the complete analysis pipeline."""
    print("Starting comprehensive fetal health analysis...")
    total_start_time = time.time()
    
    # Execute the full pipeline
    setup_environment(output_path)
    load_and_preprocess_data(data_path)
    explore_data()
    create_visualizations()
    prepare_data_for_modeling()
    
    # Show SMOTE effect before training models
    visualize_before_after_smote()
    
    train_models()
    evaluate_models()
    create_evaluation_plots()
    generate_summary_report()
    
    # Create comprehensive PDF report
    create_comprehensive_pdf_report()
    
    total_time = time.time() - total_start_time
    print(f"\nComplete analysis finished in {total_time:.2f} seconds")
    print(f"All results saved to: {save_path}")

def run_smote_analysis_only(data_path="fetal_health.csv", output_path="fetal_health_results"):
    """Run only the SMOTE before/after analysis without full model training."""
    print("Starting SMOTE analysis...")
    
    # Execute minimal pipeline for SMOTE analysis
    setup_environment(output_path)
    load_and_preprocess_data(data_path)
    prepare_data_for_modeling()
    
    # Show SMOTE effect
    visualize_before_after_smote()
    
    print(f"SMOTE analysis completed. Results saved to: {save_path}")

def demo_smote_visualization():
    """Quick demo function to show SMOTE visualization functionality."""
    print("="*60)
    print("DEMO: SMOTE Before/After Visualization")
    print("="*60)
    print("This function demonstrates the SMOTE visualization feature.")
    print("It will load data, prepare it for modeling, and show before/after SMOTE plots.")
    print()
    
    try:
        run_smote_analysis_only()
        print("\nSMOTE visualization demo completed successfully!")
        print("Check the 'fetal_health_results' folder for the generated plot:")
        print("  - before_after_smote_analysis.png")
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        print("Make sure the 'fetal_health.csv' file is in the current directory.")

def main():
    """Main function to run the fetal health prediction analysis."""
    # Configuration
    data_path = "fetal_health.csv"
    output_path = "fetal_health_results"
    
    # Run the complete analysis
    run_complete_analysis(data_path, output_path)
    
    # Additional usage examples
    print("\n" + "="*50)
    print("USAGE EXAMPLES:")
    print("="*50)
    
    print("\n1. To use a specific trained model for prediction:")
    print("   model = joblib.load('fetal_health_results/random_forest_model.joblib')")
    print("   prediction = model.predict(new_data)")
    
    print("\n2. To access the best performing model:")
    if evaluation_results:
        best_model_name = max(evaluation_results.keys(), 
                             key=lambda x: evaluation_results[x]['f1_score'])
        print(f"   best_model = models['{best_model_name}']")
    
    print("\n3. To get detailed results:")
    print("   # evaluation_results contains all model performance metrics")
    print("   # models contains the trained model objects")
    
    print("\n4. To run only SMOTE analysis (without model training):")
    print("   run_smote_analysis_only('fetal_health.csv', 'output_folder')")
    
    print("\n5. To generate only the PDF report from existing results:")
    print("   generate_pdf_from_existing_results('fetal_health.csv', 'fetal_health_results')")
    
    print("\n6. Files generated include:")
    print("   - fetal_health_analysis_report.pdf: Comprehensive PDF report with all plots")
    print("   - before_after_smote_analysis.png: SMOTE comparison visualization")
    print("   - comprehensive_data_analysis.png: Complete dataset overview")
    print("   - model_evaluation.png: Individual model performance")
    print("   - model_comparison.png: Comparative analysis")
    print("   - analysis_report.txt: Summary text report")

if __name__ == "__main__":
    main() 