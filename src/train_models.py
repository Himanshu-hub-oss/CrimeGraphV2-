"""
CrimeGraph AI — Training & Validation Pipeline
Trains the Risk Classifier, Graph Link Predictor, and NLP Threat Analyzer.
"""
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processor import DataProcessor
from src.graph_engine import CriminalGraphEngine
from src.ml_models import MLPredictorSuite

def run_training():
    print("==================================================================")
    print(" [*] Starting CrimeGraph AI Model Training Pipeline")
    print("==================================================================")
    
    # 1. Load Data
    dp = DataProcessor()
    kpis = dp.get_kpis()
    print(f"[*] Ingested Data: {kpis['total_suspects']} Suspects, {kpis['total_firs']} FIRs, {kpis['total_calls']} CDR Logs, {kpis['total_network_edges']} Network Edges")

    # 2. Build Criminal Network Graph
    print("\n[*] Initializing Network Graph Engine...")
    graph_engine = CriminalGraphEngine(dp)
    print(f"[*] Criminal Graph Built: {graph_engine.G.number_of_nodes()} Nodes, {graph_engine.G.number_of_edges()} Edges")
    
    top_kingpins = graph_engine.get_top_kingpins(top_n=5)
    print("\n[*] Top 5 Network Kingpins Identified:")
    for idx, row in top_kingpins.iterrows():
        print(f"    - {row['name']} ({row['suspect_id']}) | Gang: {row['gang_affiliation']} | Kingpin Score: {row['kingpin_score']}/100 | Betweenness: {row['betweenness_centrality']}")

    # 3. Train AI Models
    ml_suite = MLPredictorSuite()
    
    # Model 1: Suspect Risk Level Classifier
    print("\n[+] Training Model 1: Suspect Risk Classifier (Random Forest + XAI)...")
    risk_res = ml_suite.train_risk_classifier(dp.enriched_df)
    print(f"    - Risk Model Train Accuracy: {risk_res['accuracy'] * 100:.2f}%")
    print(f"    - Top 3 Important Features: {list(risk_res['feature_importances'].items())[:3]}")

    # Model 2: Criminal Link Predictor
    print("\n[+] Training Model 2: Criminal Link Predictor (Graph Topology)...")
    link_res = ml_suite.train_link_predictor(graph_engine.G, dp.edges_df, dp.suspects_df)
    print(f"    - Link Prediction Accuracy: {link_res['link_accuracy'] * 100:.2f}% (Trained on {link_res['total_pairs_trained']} node pairs)")

    # Model 3: NLP Threat Intelligence Classifier
    print("\n[+] Training Model 3: NLP Social Media Threat Intelligence Model...")
    nlp_res = ml_suite.train_nlp_threat_classifier(dp.posts_df)
    print(f"    - NLP Threat Model Accuracy: {nlp_res['nlp_accuracy'] * 100:.2f}% (Trained on {nlp_res['total_posts_trained']} social media posts)")

    # Test Sample Inferences
    print("\n==================================================================")
    print(" [*] Running Sample Inferences on Real Suspects")
    print("==================================================================")
    
    # Test Suspect 1 (SUS0024 - Suresh Shah)
    suspect_sample = dp.enriched_df[dp.enriched_df['suspect_id'] == 'SUS0024'].iloc[0].to_dict()
    pred_risk = ml_suite.predict_suspect_risk(suspect_sample)
    print(f"[*] Suspect SUS0024 (Suresh Shah) Prediction:")
    print(f"    - Predicted Risk Level: {pred_risk['predicted_risk_level']} (Confidence: {pred_risk['confidence']*100:.1f}%)")
    print(f"    - Top XAI Feature Drivers: {[f['feature'] for f in pred_risk['xai_top_features'][:3]]}")

    # Test Link Prediction between SUS0024 and SUS0104
    pred_link = ml_suite.predict_link_probability(graph_engine.G, 'SUS0024', 'SUS0104', dp.suspects_df)
    print(f"\n[*] Link Prediction between SUS0024 and SUS0104:")
    print(f"    - Link Probability: {pred_link['link_probability']*100:.1f}%")
    print(f"    - Threat Status: {pred_link['threat_status']}")

    # Test NLP on Threat Text
    test_text = "New shipment arriving from border tonight, lay low near highway naka"
    nlp_eval = ml_suite.analyze_social_post_threat(test_text)
    print(f"\n[*] NLP Threat Analysis on intercept: \"{test_text}\"")
    print(f"    - Category: {nlp_eval['threat_category']} (Alert: {nlp_eval['alert_level']})")
    print(f"    - Trigger Keywords: {nlp_eval['detected_keywords']}")

    print("\n==================================================================")
    print(" [SUCCESS] All CrimeGraph AI Models Successfully Trained and Saved to models/")
    print("==================================================================")

if __name__ == "__main__":
    run_training()
