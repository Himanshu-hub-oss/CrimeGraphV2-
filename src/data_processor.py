import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

class DataProcessor:
    """
    Data Ingestion and Preprocessing Engine for CrimeGraph AI.
    Loads, cleans, links, and aggregates multi-source criminal intelligence data.
    """
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.suspects_df = None
        self.fir_df = None
        self.cdr_df = None
        self.edges_df = None
        self.posts_df = None
        self.enriched_df = None
        self.predictions_df = None
        self.load_all_data()

    def load_all_data(self):
        """Load all CSV datasets into memory"""
        self.suspects_df = pd.read_csv(os.path.join(self.data_dir, "suspects.csv"))
        self.fir_df = pd.read_csv(os.path.join(self.data_dir, "fir_records.csv"))
        self.cdr_df = pd.read_csv(os.path.join(self.data_dir, "cdr_records.csv"))
        self.edges_df = pd.read_csv(os.path.join(self.data_dir, "network_edges.csv"))
        self.posts_df = pd.read_csv(os.path.join(self.data_dir, "social_media_posts.csv"))
        
        enriched_path = os.path.join(self.data_dir, "suspect_features_enriched.csv")
        if os.path.exists(enriched_path):
            self.enriched_df = pd.read_csv(enriched_path)
        else:
            self.enriched_df = self._generate_enriched_features()
            
        pred_path = os.path.join(self.data_dir, "suspect_risk_predictions.csv")
        if os.path.exists(pred_path):
            self.predictions_df = pd.read_csv(pred_path)
        else:
            self.predictions_df = pd.DataFrame()

    def _generate_enriched_features(self):
        """Fallback feature aggregation pipeline if enriched file is missing"""
        df = self.suspects_df.copy()
        
        # Calculate FIR count per suspect
        fir_counts = {}
        for _, row in self.fir_df.iterrows():
            suspects = str(row['accused_suspect_ids']).split(';')
            for s in suspects:
                s = s.strip()
                fir_counts[s] = fir_counts.get(s, 0) + 1
                
        df['fir_count'] = df['suspect_id'].map(lambda s: fir_counts.get(s, 0))
        
        # Calculate CDR call count
        call_counts = {}
        for _, row in self.cdr_df.iterrows():
            c1, c2 = row['caller_id'], row['callee_id']
            call_counts[c1] = call_counts.get(c1, 0) + 1
            call_counts[c2] = call_counts.get(c2, 0) + 1
        df['call_count'] = df['suspect_id'].map(lambda s: call_counts.get(s, 0))
        
        # Social media suspicious posts
        suspicious_counts = {}
        for _, row in self.posts_df.iterrows():
            if str(row['ai_sentiment_flag']).lower() in ['suspicious', 'threatening', 'coded/ambiguous']:
                s = row['suspect_id']
                suspicious_counts[s] = suspicious_counts.get(s, 0) + 1
        df['suspicious_post_count'] = df['suspect_id'].map(lambda s: suspicious_counts.get(s, 0))
        
        return df

    def get_kpis(self):
        """Compute high-level executive statistics"""
        total_suspects = len(self.suspects_df)
        active_gangs = self.suspects_df['gang_affiliation'].nunique()
        critical_risk = len(self.suspects_df[self.suspects_df['risk_level'] == 'Critical'])
        high_risk = len(self.suspects_df[self.suspects_df['risk_level'] == 'High'])
        total_firs = len(self.fir_df)
        total_calls = len(self.cdr_df)
        total_edges = len(self.edges_df)
        threat_posts = len(self.posts_df[self.posts_df['ai_sentiment_flag'].isin(['Threatening', 'Suspicious', 'Coded/Ambiguous'])])
        
        return {
            "total_suspects": total_suspects,
            "active_gangs": active_gangs,
            "critical_risk": critical_risk,
            "high_risk": high_risk,
            "total_firs": total_firs,
            "total_calls": total_calls,
            "total_network_edges": total_edges,
            "flagged_threat_posts": threat_posts
        }

    def get_gang_summary(self):
        """Summarize suspect distribution, risk profiles, and top crime types per gang"""
        gang_stats = []
        for gang, group in self.suspects_df.groupby("gang_affiliation"):
            members_count = len(group)
            critical_count = len(group[group['risk_level'] == 'Critical'])
            high_count = len(group[group['risk_level'] == 'High'])
            primary_crimes = group['primary_crime_type'].mode().tolist()
            top_crime = primary_crimes[0] if primary_crimes else "Various"
            avg_prior_cases = round(group['prior_cases_count'].mean(), 1)
            
            # Find Kingpins in this gang
            kingpins = group[group['role_in_network'] == 'Kingpin']['name'].tolist()
            enforcers = group[group['role_in_network'] == 'Enforcer']['name'].tolist()
            financiers = group[group['role_in_network'] == 'Financier']['name'].tolist()
            
            gang_stats.append({
                "gang_id": gang,
                "members_count": members_count,
                "critical_suspects": critical_count,
                "high_suspects": high_count,
                "top_crime_type": top_crime,
                "avg_prior_cases": avg_prior_cases,
                "kingpins": ", ".join(kingpins) if kingpins else "None Identified",
                "enforcers_count": len(enforcers),
                "financiers_count": len(financiers)
            })
            
        return pd.DataFrame(gang_stats).sort_values(by="critical_suspects", ascending=False)

    def get_city_crime_hotspots(self):
        """Aggregate crime counts and CDR activity by city for heatmaps"""
        fir_city = self.fir_df.groupby("city").size().reset_index(name="fir_count")
        suspect_city = self.suspects_df.groupby("city").size().reset_index(name="suspect_count")
        cdr_city = self.cdr_df.groupby("cell_tower_city").size().reset_index(name="cdr_activity_count").rename(columns={"cell_tower_city": "city"})
        
        merged = pd.merge(fir_city, suspect_city, on="city", how="outer").fillna(0)
        merged = pd.merge(merged, cdr_city, on="city", how="outer").fillna(0)
        
        # Approximate latitude and longitude coordinates for Indian cities in the dataset
        city_coords = {
            "Delhi": {"lat": 28.6139, "lon": 77.2090},
            "Mumbai": {"lat": 19.0760, "lon": 72.8777},
            "Pune": {"lat": 18.5204, "lon": 73.8567},
            "Jaipur": {"lat": 26.9124, "lon": 75.7873},
            "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
            "Surat": {"lat": 21.1702, "lon": 72.8311},
            "Nagpur": {"lat": 21.1458, "lon": 79.0882},
            "Indore": {"lat": 22.7196, "lon": 75.8577},
            "Bhopal": {"lat": 23.2599, "lon": 77.4126},
            "Lucknow": {"lat": 26.8467, "lon": 80.9462},
            "Kanpur": {"lat": 26.4499, "lon": 80.3319},
            "Patna": {"lat": 25.5941, "lon": 85.1376},
            "Ranchi": {"lat": 23.3441, "lon": 85.3096},
            "Amritsar": {"lat": 31.6340, "lon": 74.8723},
            "Chandigarh": {"lat": 30.7333, "lon": 76.7794}
        }
        
        merged['lat'] = merged['city'].map(lambda c: city_coords.get(c, {}).get("lat", 20.5937))
        merged['lon'] = merged['city'].map(lambda c: city_coords.get(c, {}).get("lon", 78.9629))
        
        # Calculate Composite Crime Severity Score
        merged['severity_score'] = (merged['fir_count'] * 3 + merged['suspect_count'] * 2 + merged['cdr_activity_count'] * 0.1).round(1)
        return merged.sort_values(by="severity_score", ascending=False)

    def get_suspect_dossier(self, suspect_id):
        """
        Compile an exhaustive legal/intelligence dossier for a suspect,
        including demographic details, FIR cases, co-accused accomplices,
        CDR call statistics, social media footprint, and network position.
        """
        suspect_row = self.suspects_df[self.suspects_df['suspect_id'] == suspect_id]
        if suspect_row.empty:
            return None
            
        profile = suspect_row.iloc[0].to_dict()
        
        # Enriched features if available
        enriched_row = self.enriched_df[self.enriched_df['suspect_id'] == suspect_id]
        if not enriched_row.empty:
            profile.update(enriched_row.iloc[0].to_dict())
            
        # Linked FIRs
        linked_firs = []
        co_accused_ids = set()
        for _, row in self.fir_df.iterrows():
            accused_list = [s.strip() for s in str(row['accused_suspect_ids']).split(';')]
            if suspect_id in accused_list:
                linked_firs.append(row.to_dict())
                for other in accused_list:
                    if other != suspect_id:
                        co_accused_ids.add(other)
                        
        profile['linked_firs'] = linked_firs
        profile['co_accused_count'] = len(co_accused_ids)
        profile['co_accused_ids'] = list(co_accused_ids)
        
        # Linked CDR Calls
        caller_calls = self.cdr_df[self.cdr_df['caller_id'] == suspect_id]
        callee_calls = self.cdr_df[self.cdr_df['callee_id'] == suspect_id]
        
        total_calls = len(caller_calls) + len(callee_calls)
        unique_contacts = set(caller_calls['callee_id'].tolist() + callee_calls['caller_id'].tolist())
        unique_contacts.discard(suspect_id)
        
        profile['cdr_total_calls'] = total_calls
        profile['unique_contacts_count'] = len(unique_contacts)
        profile['top_contacts'] = list(unique_contacts)[:10]
        profile['recent_calls'] = pd.concat([caller_calls, callee_calls]).sort_values(by="timestamp", ascending=False).head(10).to_dict('records')
        
        # Linked Social Media Posts
        suspect_posts = self.posts_df[self.posts_df['suspect_id'] == suspect_id]
        profile['social_posts'] = suspect_posts.to_dict('records')
        profile['threat_posts_count'] = len(suspect_posts[suspect_posts['ai_sentiment_flag'].isin(['Threatening', 'Suspicious', 'Coded/Ambiguous'])])
        
        # Linked Network Edges
        connected_edges = self.edges_df[(self.edges_df['source_suspect_id'] == suspect_id) | (self.edges_df['target_suspect_id'] == suspect_id)]
        profile['network_edges'] = connected_edges.to_dict('records')
        
        return profile
