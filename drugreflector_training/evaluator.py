"""
Model evaluator for DrugReflector ensemble.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from .preprocessing import clip_and_normalize_signature


class DrugReflectorEvaluator:
    """
    Evaluator for DrugReflector ensemble models.
    
    Parameters
    ----------
    models : List[nn.Module]
        List of trained models
    device : str, default='cuda'
        Device for computation
    """
    
    def __init__(self, models: List[torch.nn.Module], device='cuda'):
        self.models = models
        self.device = device
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions by averaging model scores.
        
        From SI Page 2:
        "The final predicted class probabilities were the softmax probabilities 
        of the average score over all three folds."
        
        Parameters
        ----------
        X : np.ndarray
            Input data (n_samples, 978)
        
        Returns
        -------
        np.ndarray
            Average scores (n_samples, n_compounds)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        all_scores = []
        with torch.no_grad():
            for model in self.models:
                scores = model(X_tensor)
                all_scores.append(scores.cpu().numpy())
        
        # Average scores across models (NOT probabilities)
        avg_scores = np.mean(all_scores, axis=0)
        
        return avg_scores
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        compound_names: List[str]
    ) -> Dict:
        """
        Evaluate ensemble performance on test data.
        
        Parameters
        ----------
        X : np.ndarray
            Test data (n_samples, 978)
        y : np.ndarray
            True labels (n_samples,)
        compound_names : List[str]
            List of compound names
        
        Returns
        -------
        Dict
            Evaluation metrics
        """
        print(f"\n📊 Evaluating ensemble performance...")
        
        # Preprocess
        X_processed = clip_and_normalize_signature(X)
        
        # Predict
        avg_scores = self.predict_ensemble(X_processed)
        probs = torch.softmax(torch.FloatTensor(avg_scores), dim=1).numpy()
        preds = np.argmax(avg_scores, axis=1)
        
        # Compute metrics
        top1_acc = accuracy_score(y, preds)
        top10_acc = top_k_accuracy_score(y, probs, k=10)
        
        # Top 1% recall (main metric in paper)
        top1_percent_k = max(1, int(0.01 * probs.shape[1]))
        recall = top_k_accuracy_score(y, probs, k=top1_percent_k)
        
        print(f"  Top-1 Accuracy: {top1_acc:.4f}")
        print(f"  Top-10 Accuracy: {top10_acc:.4f}")
        print(f"  Top 1% Recall: {recall:.4f}")
        
        # Per-compound recall
        compound_recalls = []
        for compound_idx in range(len(compound_names)):
            mask = y == compound_idx
            if mask.sum() == 0:
                continue
            
            compound_probs = probs[mask]
            compound_labels = y[mask]
            
            if len(compound_probs) > 0:
                compound_recall = top_k_accuracy_score(
                    compound_labels, 
                    compound_probs, 
                    k=top1_percent_k
                )
                compound_recalls.append(compound_recall)
        
        avg_compound_recall = np.mean(compound_recalls) if compound_recalls else 0.0
        print(f"  Average per-compound recall: {avg_compound_recall:.4f}")
        
        results = {
            'top1_accuracy': top1_acc,
            'top10_accuracy': top10_acc,
            'top1_percent_recall': recall,
            'avg_compound_recall': avg_compound_recall,
            'compound_recalls': compound_recalls,
            'predictions': preds,
            'probabilities': probs,
            'average_scores': avg_scores
        }
        
        return results