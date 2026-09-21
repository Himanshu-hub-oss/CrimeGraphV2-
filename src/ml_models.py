import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import networkx as nx

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class MLPredictorSuite:
    """
    AI/ML Prediction Suite for CrimeGraph AI:
    - Model 1: Suspect Risk Level Classifier & XAI Explainer
    - Model 2: Graph-based Criminal Link Predictor
    - Model 3: NLP Social Media Threat Intelligence Classifier
    """
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.risk_model = None
        self.risk_scaler = None
        self.risk_encoder = None
        self.risk_feature_cols = None
        
        self.link_model = None
        self.link_scaler = None
        
        self.nlp_vectorizer = None
        self.nlp_model = None
        self.nlp_encoder = None
        
        self.is_trained = False
        self.load_models()

    def train_risk_classifier(self, enriched_df):
        """Train Random Forest multi-class Risk Classifier on Suspect Enriched Features"""
        feature_cols = [
            'age', 'prior_cases_count', 'fir_count', 'degree', 'weighted_degree',
            'clustering_coeff', 'betweenness', 'eigenvector_centrality',
            'call_count', 'total_sent_inr', 'total_received_inr',
            'flagged_txn_count', 'suspicious_post_count'
        ]
        self.risk_feature_cols = feature_cols
        
        # Clean missing values
        df = enriched_df.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        X = df[feature_cols].values
        y_raw = df['risk_level'].astype(str).values

        self.risk_encoder = LabelEncoder()
        y = self.risk_encoder.fit_transform(y_raw)

        self.risk_scaler = StandardScaler()
        X_scaled = self.risk_scaler.fit_transform(X)

        self.risk_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        self.risk_model.fit(X_scaled, y)

        # Feature importances
        importances = dict(zip(feature_cols, [round(float(v), 4) for v in self.risk_model.feature_importances_]))
        
        # Save model
        joblib.dump(self.risk_model, os.path.join(self.models_dir, "risk_model.joblib"))
        joblib.dump(self.risk_scaler, os.path.join(self.models_dir, "risk_scaler.joblib"))
        joblib.dump(self.risk_encoder, os.path.join(self.models_dir, "risk_encoder.joblib"))
        joblib.dump(self.risk_feature_cols, os.path.join(self.models_dir, "risk_feature_cols.joblib"))

        train_acc = accuracy_score(y, self.risk_model.predict(X_scaled))
        return {"accuracy": round(train_acc, 4), "feature_importances": importances}

    def predict_suspect_risk(self, suspect_features_dict):
        """
        Predict risk level (Low, Medium, High, Critical), probability distribution,
        and top contributing features (Explainable AI / XAI).
        """
        if self.risk_model is None:
            self.load_models()
            if self.risk_model is None:
                return {"error": "Risk Model not trained"}

        vals = []
        for col in self.risk_feature_cols:
            vals.append(float(suspect_features_dict.get(col, 0.0)))

        X_input = np.array([vals])
        X_scaled = self.risk_scaler.transform(X_input)

        probs = self.risk_model.predict_proba(X_scaled)[0]
        classes = self.risk_encoder.classes_
        
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
        pred_class = classes[np.argmax(probs)]
        confidence = round(float(np.max(probs)), 4)

        # Explainable AI (XAI) feature contribution estimate
        importances = self.risk_model.feature_importances_
        feature_contributions = []
        for col, val, imp in zip(self.risk_feature_cols, vals, importances):
            impact = round(float(val * imp), 4)
            feature_contributions.append({
                "feature": col,
                "value": val,
                "importance_weight": round(float(imp), 4),
                "impact_score": impact
            })
            
        feature_contributions.sort(key=lambda x: x['impact_score'], reverse=True)

        return {
            "predicted_risk_level": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "xai_top_features": feature_contributions[:6]
        }

    def train_link_predictor(self, graph, edges_df, suspects_df):
        """
        Train link prediction model using topological indices (Adamic-Adar, Jaccard, Common Neighbors)
        and behavioral attributes to predict hidden collusions.
        """
        suspect_meta = suspects_df.set_index('suspect_id').to_dict('index')
        
        positive_pairs = []
        for _, row in edges_df.iterrows():
            u, v = row['source_suspect_id'], row['target_suspect_id']
            if graph.has_node(u) and graph.has_node(v):
                positive_pairs.append((u, v, 1))

        # Sample negative pairs (nodes with no direct edge)
        all_nodes = list(graph.nodes())
        negative_pairs = []
        rng = np.random.RandomState(42)
        
        while len(negative_pairs) < len(positive_pairs) and len(all_nodes) > 1:
            u, v = rng.choice(all_nodes, 2, replace=False)
            if not graph.has_edge(u, v) and (v, u) not in [p[:2] for p in positive_pairs]:
                negative_pairs.append((u, v, 0))

        dataset = positive_pairs + negative_pairs
        X, y = [], []

        for u, v, label in dataset:
            features = self._extract_pair_features(graph, u, v, suspect_meta)
            X.append(features)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        self.link_scaler = StandardScaler()
        X_scaled = self.link_scaler.fit_transform(X)

        self.link_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.link_model.fit(X_scaled, y)

        joblib.dump(self.link_model, os.path.join(self.models_dir, "link_model.joblib"))
        joblib.dump(self.link_scaler, os.path.join(self.models_dir, "link_scaler.joblib"))

        train_acc = accuracy_score(y, self.link_model.predict(X_scaled))
        return {"link_accuracy": round(train_acc, 4), "total_pairs_trained": len(dataset)}

    def _extract_pair_features(self, graph, u, v, suspect_meta):
        """Extract topological & attribute similarity features between two suspects"""
        if not graph.has_node(u) or not graph.has_node(v):
            return [0.0] * 8

        # Topological graph features
        common_nbrs = len(list(nx.common_neighbors(graph, u, v)))
        
        # Jaccard Coefficient
        nbrs_u = set(graph.neighbors(u))
        nbrs_v = set(graph.neighbors(v))
        union_len = len(nbrs_u.union(nbrs_v))
        jaccard = (common_nbrs / union_len) if union_len > 0 else 0.0

        # Adamic-Adar Index
        adamic_adar = 0.0
        for w in nbrs_u.intersection(nbrs_v):
            deg = graph.degree(w)
            if deg > 1:
                adamic_adar += 1.0 / np.log(deg)

        # Preferential Attachment
        pref_attachment = float(graph.degree(u) * graph.degree(v))

        # Node attribute features
        meta_u = suspect_meta.get(u, {})
        meta_v = suspect_meta.get(v, {})

        same_gang = 1.0 if meta_u.get('gang_affiliation') == meta_v.get('gang_affiliation') and meta_u.get('gang_affiliation') else 0.0
        same_city = 1.0 if meta_u.get('city') == meta_v.get('city') and meta_u.get('city') else 0.0
        same_crime = 1.0 if meta_u.get('primary_crime_type') == meta_v.get('primary_crime_type') and meta_u.get('primary_crime_type') else 0.0
        prior_cases_sum = float(meta_u.get('prior_cases_count', 0) + meta_v.get('prior_cases_count', 0))

        return [common_nbrs, jaccard, adamic_adar, pref_attachment, same_gang, same_city, same_crime, prior_cases_sum]

    def predict_link_probability(self, graph, suspect_a, suspect_b, suspects_df):
        """Predict probability of criminal collusion between suspect A and suspect B"""
        if self.link_model is None:
            self.load_models()
            if self.link_model is None:
                return {"error": "Link Prediction Model not trained"}

        suspect_meta = suspects_df.set_index('suspect_id').to_dict('index')
        features = self._extract_pair_features(graph, suspect_a, suspect_b, suspect_meta)
        
        X_input = np.array([features])
        X_scaled = self.link_scaler.transform(X_input)

        probs = self.link_model.predict_proba(X_scaled)[0]
        link_prob = round(float(probs[1]), 4) if len(probs) > 1 else 0.5

        existing_edge = graph.has_edge(suspect_a, suspect_b)
        
        if link_prob >= 0.70:
            threat_status = "🚨 High Criminal Collusion Probability"
        elif link_prob >= 0.40:
            threat_status = "⚠️ Moderate Indirect Link Potential"
        else:
            threat_status = "🟢 Low Probability of Collusion"

        return {
            "suspect_a": suspect_a,
            "suspect_b": suspect_b,
            "link_probability": link_prob,
            "threat_status": threat_status,
            "direct_edge_exists": existing_edge,
            "common_accomplices_count": int(features[0]),
            "jaccard_similarity": round(float(features[1]), 4),
            "adamic_adar_score": round(float(features[2]), 4),
            "same_gang": bool(features[4]),
            "same_city": bool(features[5])
        }

    def train_nlp_threat_classifier(self, posts_df):
        """Train TF-IDF + Classifier on Social Media Posts for Threat Classification"""
        df = posts_df.dropna(subset=['post_text', 'ai_sentiment_flag']).copy()
        
        X_text = df['post_text'].astype(str).tolist()
        y_raw = df['ai_sentiment_flag'].astype(str).tolist()

        self.nlp_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, sublinear_tf=True)
        X_tfidf = self.nlp_vectorizer.fit_transform(X_text)

        self.nlp_encoder = LabelEncoder()
        y = self.nlp_encoder.fit_transform(y_raw)

        self.nlp_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.nlp_model.fit(X_tfidf, y)

        joblib.dump(self.nlp_model, os.path.join(self.models_dir, "nlp_model.joblib"))
        joblib.dump(self.nlp_vectorizer, os.path.join(self.models_dir, "nlp_vectorizer.joblib"))
        joblib.dump(self.nlp_encoder, os.path.join(self.models_dir, "nlp_encoder.joblib"))

        train_acc = accuracy_score(y, self.nlp_model.predict(X_tfidf))
        return {"nlp_accuracy": round(train_acc, 4), "total_posts_trained": len(df)}

    def analyze_social_post_threat(self, text):
        """Classify given text into Neutral, Suspicious, Threatening, or Coded/Ambiguous"""
        if self.nlp_model is None:
            self.load_models()
            if self.nlp_model is None:
                return {"error": "NLP Model not trained"}

        X_tfidf = self.nlp_vectorizer.transform([str(text)])
        probs = self.nlp_model.predict_proba(X_tfidf)[0]
        classes = self.nlp_encoder.classes_

        pred_class = classes[np.argmax(probs)]
        confidence = round(float(np.max(probs)), 4)
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}

        # Keyword trigger detection
        triggers = []
        keywords = ["shipment", "border", "package", "delivered", "naka", "police", "lay low", "hot", "usual spot", "payment cleared", "big deal"]
        for kw in keywords:
            if kw in str(text).lower():
                triggers.append(kw)

        return {
            "threat_category": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "detected_keywords": triggers,
            "alert_level": "CRITICAL" if pred_class in ["Threatening", "Suspicious"] else "NORMAL"
        }

    def load_models(self):
        """Load trained models from disk if present"""
        try:
            r_path = os.path.join(self.models_dir, "risk_model.joblib")
            if os.path.exists(r_path):
                self.risk_model = joblib.load(r_path)
                self.risk_scaler = joblib.load(os.path.join(self.models_dir, "risk_scaler.joblib"))
                self.risk_encoder = joblib.load(os.path.join(self.models_dir, "risk_encoder.joblib"))
                self.risk_feature_cols = joblib.load(os.path.join(self.models_dir, "risk_feature_cols.joblib"))

            l_path = os.path.join(self.models_dir, "link_model.joblib")
            if os.path.exists(l_path):
                self.link_model = joblib.load(l_path)
                self.link_scaler = joblib.load(os.path.join(self.models_dir, "link_scaler.joblib"))

            n_path = os.path.join(self.models_dir, "nlp_model.joblib")
            if os.path.exists(n_path):
                self.nlp_model = joblib.load(n_path)
                self.nlp_vectorizer = joblib.load(os.path.join(self.models_dir, "nlp_vectorizer.joblib"))
                self.nlp_encoder = joblib.load(os.path.join(self.models_dir, "nlp_encoder.joblib"))

            if self.risk_model and self.link_model and self.nlp_model:
                self.is_trained = True
        except Exception as e:
            print(f"Error loading models: {e}")
