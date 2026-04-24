"""Evaluation module for code-mixed text classification."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
import torch

from config import DEVICE, DATA_PATHS, LABELS, ID2LABEL
from utils import setup_logging, save_json


class Evaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = DEVICE
        self.logger = setup_logging('evaluate')
    
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def compute_metrics(self, y_true, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        per_class = {}
        for label_id, label_name in ID2LABEL.items():
            y_true_bin = (y_true == label_id).astype(int)
            y_pred_bin = (y_pred == label_id).astype(int)
            per_class[label_name] = {
                'precision': precision_score(y_true_bin, y_pred_bin, zero_division=0),
                'recall': recall_score(y_true_bin, y_pred_bin, zero_division=0),
                'f1': f1_score(y_true_bin, y_pred_bin, zero_division=0),
                'support': int(np.sum(y_true == label_id))
            }
        
        metrics['per_class'] = per_class
        return metrics
    
    def generate_report(self, metrics):
        lines = ["=" * 60, "EVALUATION REPORT", "=" * 60, "",
                 f"Overall Accuracy: {metrics['accuracy']:.4f}", "",
                 "Macro-Averaged Metrics:",
                 f"  Precision: {metrics['macro_precision']:.4f}",
                 f"  Recall:    {metrics['macro_recall']:.4f}",
                 f"  F1-Score:  {metrics['macro_f1']:.4f}", "",
                 "Weighted-Averaged Metrics:",
                 f"  Precision: {metrics['weighted_precision']:.4f}",
                 f"  Recall:    {metrics['weighted_recall']:.4f}",
                 f"  F1-Score:  {metrics['weighted_f1']:.4f}", "",
                 "Per-Class Performance:", "-" * 60]
        
        for label_name, scores in metrics['per_class'].items():
            lines.extend([
                f"  {label_name}:",
                f"    Precision: {scores['precision']:.4f}",
                f"    Recall:    {scores['recall']:.4f}",
                f"    F1-Score:  {scores['f1']:.4f}",
                f"    Support:   {scores['support']}", ""
            ])
        
        report = "\n".join(lines)
        path = os.path.join(DATA_PATHS['result_dir'], 'evaluation_report.txt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(report)
        self.logger.info(f"Report saved to {path}")
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=LABELS, yticklabels=LABELS)
        plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(DATA_PATHS['result_dir'], 'confusion_matrix.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Confusion matrix saved")
    
    def plot_training_curves(self, history, save_path=None):
        if save_path is None:
            save_path = os.path.join(DATA_PATHS['result_dir'], 'training_curves.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(history['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(history['val_loss'], label='Val', marker='s')
        axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(history['val_acc'], marker='o', color='green')
        axes[0, 1].set_title('Val Accuracy'); axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(history['val_f1'], marker='o', color='purple')
        axes[1, 0].set_title('Val F1'); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(history['train_loss'], label='Train Loss', color='blue')
        ax_twin = axes[1, 1].twinx()
        ax_twin.plot(history['val_f1'], label='Val F1', color='red', marker='o')
        axes[1, 1].set_title('Loss vs F1')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_predictions(self, texts, y_true, y_pred, y_probs, save_path=None):
        if save_path is None:
            save_path = os.path.join(DATA_PATHS['result_dir'], 'predictions.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        results = []
        for text, true, pred, probs in zip(texts, y_true, y_pred, y_probs):
            results.append({
                'text': text,
                'true_label': ID2LABEL[true],
                'predicted_label': ID2LABEL[pred],
                'correct': true == pred,
                'confidence': float(probs[pred]),
                **{f'prob_{LABELS[j]}': float(probs[j]) for j in range(len(LABELS))}
            })
        
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Predictions saved")
        return df
    
    def run_full_evaluation(self, texts, history=None):
        self.logger.info("Starting evaluation...")
        y_pred, y_true, y_probs = self.evaluate()
        metrics = self.compute_metrics(y_true, y_pred)
        report = self.generate_report(metrics)
        print(report)
        self.plot_confusion_matrix(y_true, y_pred)
        if history:
            self.plot_training_curves(history)
        self.save_predictions(texts, y_true, y_pred, y_probs)
        metrics_path = os.path.join(DATA_PATHS['result_dir'], 'metrics.json')
        save_json(metrics, metrics_path)
        self.logger.info("Evaluation completed")
        return metrics
