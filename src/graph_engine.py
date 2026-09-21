import os
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
import tempfile

class CriminalGraphEngine:
    """
    Advanced Graph Analytics & Network Intelligence Engine.
    Executes Centrality Analysis, Louvain Community Detection, Kingpin Identification,
    and Shortest Path Intermediary Tracing.
    """
    def __init__(self, data_processor):
        self.dp = data_processor
        self.G = nx.Graph()
        self.build_graph()
        self.compute_all_metrics()

    def build_graph(self):
        """Construct weighted undirected criminal network graph with node and edge attributes"""
        self.G.clear()
        
        # 1. Add all suspect nodes with rich profile attributes
        for _, row in self.dp.suspects_df.iterrows():
            suspect_id = row['suspect_id']
            self.G.add_node(
                suspect_id,
                name=row['name'],
                age=int(row['age']),
                gender=row['gender'],
                city=row['city'],
                gang=row['gang_affiliation'],
                role=row['role_in_network'],
                crime_type=row['primary_crime_type'],
                risk_level=row['risk_level'],
                phone=row['phone_number'],
                prior_cases=int(row['prior_cases_count'])
            )

        # 2. Add edges from network_edges.csv
        for _, row in self.dp.edges_df.iterrows():
            u, v = row['source_suspect_id'], row['target_suspect_id']
            weight = float(row['interaction_weight'])
            edge_type = row['edge_type']
            
            if self.G.has_node(u) and self.G.has_node(v):
                if self.G.has_edge(u, v):
                    self.G[u][v]['weight'] += weight
                    self.G[u][v]['edge_types'].add(edge_type)
                else:
                    self.G.add_edge(u, v, weight=weight, edge_types={edge_type}, primary_type=edge_type)

        # 3. Add co-accused relationships from FIRs
        for _, row in self.dp.fir_df.iterrows():
            accused_list = [s.strip() for s in str(row['accused_suspect_ids']).split(';') if s.strip()]
            for i in range(len(accused_list)):
                for j in range(i + 1, len(accused_list)):
                    u, v = accused_list[i], accused_list[j]
                    if self.G.has_node(u) and self.G.has_node(v):
                        if self.G.has_edge(u, v):
                            self.G[u][v]['weight'] += 2.0
                            self.G[u][v]['edge_types'].add("Co-Accused FIR")
                        else:
                            self.G.add_edge(u, v, weight=2.0, edge_types={"Co-Accused FIR"}, primary_type="Co-Accused FIR")

        # 4. Integrate CDR call frequencies
        cdr_pairs = self.dp.cdr_df.groupby(['caller_id', 'callee_id']).size().reset_index(name='call_count')
        for _, row in cdr_pairs.iterrows():
            u, v = row['caller_id'], row['callee_id']
            calls = int(row['call_count'])
            if self.G.has_node(u) and self.G.has_node(v):
                if self.G.has_edge(u, v):
                    self.G[u][v]['weight'] += calls * 0.5
                    self.G[u][v]['edge_types'].add("CDR Call")
                else:
                    self.G.add_edge(u, v, weight=calls * 0.5, edge_types={"CDR Call"}, primary_type="CDR Call")

    def compute_all_metrics(self):
        """Calculate graph topology metrics: Centralities & Communities"""
        # Centralities
        degree_dict = dict(self.G.degree())
        weighted_degree_dict = dict(self.G.degree(weight='weight'))
        betweenness_dict = nx.betweenness_centrality(self.G, weight='weight', normalized=True)
        closeness_dict = nx.closeness_centrality(self.G)
        
        try:
            eigenvector_dict = nx.eigenvector_centrality(self.G, max_iter=1000, weight='weight')
        except Exception:
            eigenvector_dict = nx.degree_centrality(self.G)
            
        pagerank_dict = nx.pagerank(self.G, weight='weight')
        clustering_dict = nx.clustering(self.G, weight='weight')

        # Community Detection (Louvain or Greedy Modularity fallback)
        try:
            communities = list(nx.community.louvain_communities(self.G, weight='weight', seed=42))
        except Exception:
            communities = list(nx.community.greedy_modularity_communities(self.G, weight='weight'))

        community_map = {}
        for comm_id, members in enumerate(communities):
            for node in members:
                community_map[node] = comm_id

        # Update node attributes & Kingpin Score
        for node in self.G.nodes():
            deg = degree_dict.get(node, 0)
            w_deg = round(weighted_degree_dict.get(node, 0), 2)
            btw = betweenness_dict.get(node, 0.0)
            cls_c = closeness_dict.get(node, 0.0)
            eig = eigenvector_dict.get(node, 0.0)
            pr = pagerank_dict.get(node, 0.0)
            clust = clustering_dict.get(node, 0.0)
            comm_id = community_map.get(node, 0)

            # Composite Kingpin Influence Score (Normalized 0 to 100)
            # Combines network bottleneck role (betweenness), power prestige (eigenvector), and connectivity (degree)
            prior_cases = self.G.nodes[node].get('prior_cases', 0)
            kingpin_score = round(
                (btw * 35.0) + (eig * 35.0) + (pr * 15.0) + (min(deg, 20) / 20.0 * 10.0) + (min(prior_cases, 6) / 6.0 * 5.0) * 100, 
                1
            )
            kingpin_score = min(max(kingpin_score, 5.0), 99.9)

            self.G.nodes[node]['degree'] = deg
            self.G.nodes[node]['weighted_degree'] = w_deg
            self.G.nodes[node]['betweenness'] = round(btw, 5)
            self.G.nodes[node]['closeness'] = round(cls_c, 5)
            self.G.nodes[node]['eigenvector'] = round(eig, 5)
            self.G.nodes[node]['pagerank'] = round(pr, 5)
            self.G.nodes[node]['clustering_coeff'] = round(clust, 4)
            self.G.nodes[node]['community_id'] = comm_id
            self.G.nodes[node]['kingpin_score'] = kingpin_score

    def get_top_kingpins(self, top_n=15):
        """Identify top influential network kingpins based on centrality and structural topology"""
        records = []
        for node, data in self.G.nodes(data=True):
            records.append({
                "suspect_id": node,
                "name": data['name'],
                "gang_affiliation": data['gang'],
                "role": data['role'],
                "primary_crime": data['crime_type'],
                "risk_level": data['risk_level'],
                "kingpin_score": data.get('kingpin_score', 0),
                "betweenness_centrality": data.get('betweenness', 0),
                "eigenvector_centrality": data.get('eigenvector', 0),
                "degree": data.get('degree', 0),
                "prior_cases": data.get('prior_cases', 0),
                "city": data.get('city', 'Unknown')
            })
            
        df = pd.DataFrame(records).sort_values(by="kingpin_score", ascending=False)
        return df.head(top_n)

    def find_shortest_criminal_path(self, source_id, target_id):
        """
        Find shortest criminal path and intermediary conspirators between two suspects.
        Returns path nodes, degree of separation, and interaction details along each hop.
        """
        if not self.G.has_node(source_id) or not self.G.has_node(target_id):
            return {"found": False, "message": "One or both suspect IDs not found in network"}

        if not nx.has_path(self.G, source_id, target_id):
            return {"found": False, "message": "No connecting network path found between these suspects"}

        path = nx.shortest_path(self.G, source=source_id, target=target_id, weight=None)
        degrees_of_separation = len(path) - 1

        hops = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.G[u][v]
            edge_types = list(edge_data.get('edge_types', [])) if isinstance(edge_data.get('edge_types'), (set, list)) else [edge_data.get('primary_type', 'Contact')]
            hops.append({
                "from_id": u,
                "from_name": self.G.nodes[u]['name'],
                "to_id": v,
                "to_name": self.G.nodes[v]['name'],
                "weight": edge_data.get('weight', 1),
                "connection_types": ", ".join(edge_types)
            })

        return {
            "found": True,
            "path": path,
            "degrees_of_separation": degrees_of_separation,
            "hops": hops,
            "intermediary_count": max(0, len(path) - 2)
        }

    def get_mutual_accomplices(self, suspect_a, suspect_b):
        """Find common co-conspirators / mutual neighbors between two suspects"""
        if not self.G.has_node(suspect_a) or not self.G.has_node(suspect_b):
            return []
        
        neighbors_a = set(self.G.neighbors(suspect_a))
        neighbors_b = set(self.G.neighbors(suspect_b))
        common = neighbors_a.intersection(neighbors_b)
        
        results = []
        for c in common:
            data = self.G.nodes[c]
            results.append({
                "suspect_id": c,
                "name": data['name'],
                "gang": data['gang'],
                "role": data['role'],
                "risk_level": data['risk_level'],
                "kingpin_score": data.get('kingpin_score', 0)
            })
        return sorted(results, key=lambda x: x['kingpin_score'], reverse=True)

    def generate_pyvis_html(self, filter_gang=None, filter_risk=None, highlight_id=None, height="650px"):
        """
        Generate interactive, physics-enabled Pyvis network HTML graph
        with color-coded nodes by gang/risk and sized by Kingpin centrality.
        """
        net = Network(height=height, width="100%", bgcolor="#0E1117", font_color="#FFFFFF")
        net.force_atlas_2based(gravity=-60, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.95)

        # Gang Color Palette
        gang_colors = {
            "GANG01": "#FF4B4B", "GANG02": "#FFAA00", "GANG03": "#00CC96",
            "GANG04": "#AB63FA", "GANG05": "#19D3F3", "GANG06": "#FF6692",
            "GANG07": "#B6E880", "GANG08": "#FF97FF", "GANG09": "#FECB52",
            "GANG10": "#00F0FF", "GANG11": "#FF007F", "GANG12": "#7F00FF"
        }
        
        risk_colors = {
            "Critical": "#FF0055",
            "High": "#FF7700",
            "Medium": "#FFCC00",
            "Low": "#00DD88"
        }

        # Filter nodes
        filtered_nodes = set()
        for node, data in self.G.nodes(data=True):
            if filter_gang and filter_gang != "All" and data.get('gang') != filter_gang:
                continue
            if filter_risk and filter_risk != "All" and data.get('risk_level') != filter_risk:
                continue
            filtered_nodes.add(node)

        # If filtered set is too small or specific highlight is selected, add its neighbors
        if highlight_id and highlight_id in self.G:
            filtered_nodes.add(highlight_id)
            for nbr in self.G.neighbors(highlight_id):
                filtered_nodes.add(nbr)

        # Add Nodes to Pyvis
        for node in filtered_nodes:
            data = self.G.nodes[node]
            gang = data.get('gang', 'GANG01')
            risk = data.get('risk_level', 'Medium')
            
            node_color = risk_colors.get(risk, "#00CC96") if filter_risk != "All" and filter_risk is not None else gang_colors.get(gang, "#00CC96")
            
            if highlight_id and node == highlight_id:
                node_color = "#FFFFFF"
                size = 35
                border_width = 4
            else:
                size = 12 + (data.get('kingpin_score', 10) / 100.0) * 22
                border_width = 2

            title_html = (
                f"<b>{data['name']} ({node})</b><br>"
                f"<b>Gang:</b> {gang}<br>"
                f"<b>Role:</b> {data['role']}<br>"
                f"<b>Risk Level:</b> {risk}<br>"
                f"<b>Kingpin Score:</b> {data.get('kingpin_score', 0)}/100<br>"
                f"<b>Betweenness:</b> {data.get('betweenness', 0)}<br>"
                f"<b>Prior Cases:</b> {data.get('prior_cases', 0)}<br>"
                f"<b>City:</b> {data.get('city', 'Unknown')}"
            )

            net.add_node(
                node,
                label=f"{data['name']}\n[{node}]",
                title=title_html,
                color=node_color,
                size=size,
                borderWidth=border_width
            )

        # Add Edges to Pyvis
        for u, v, data in self.G.edges(data=True):
            if u in filtered_nodes and v in filtered_nodes:
                weight = float(data.get('weight', 1.0))
                primary_type = data.get('primary_type', 'Call')
                
                edge_color = "#4A5568"
                if "Financial" in str(primary_type):
                    edge_color = "#FFD700"
                elif "FIR" in str(primary_type):
                    edge_color = "#FF4B4B"

                width = min(max(1, int(weight / 2)), 6)
                net.add_edge(u, v, value=weight, title=f"Type: {primary_type} | Weight: {weight}", color=edge_color, width=width)

        # Save HTML to a temporary file
        temp_dir = tempfile.gettempdir()
        html_path = os.path.join(temp_dir, "crimegraph_network.html")
        net.save_graph(html_path)
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        return html_content

    def generate_plotly_3d_graph(self, filter_gang=None, top_n=80):
        """Generate high-tech 3D Cyber Criminal Network in Plotly"""
        sub_nodes = [n for n, d in self.G.nodes(data=True) if (filter_gang in [None, "All"] or d['gang'] == filter_gang)]
        if len(sub_nodes) > top_n:
            # Pick highest centrality nodes
            sub_nodes = sorted(sub_nodes, key=lambda n: self.G.nodes[n].get('kingpin_score', 0), reverse=True)[:top_n]
            
        sub_g = self.G.subgraph(sub_nodes)
        pos = nx.spring_layout(sub_g, dim=3, seed=42)

        # Edge traces
        edge_x, edge_y, edge_z = [], [], []
        for u, v in sub_g.edges():
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(100, 150, 255, 0.25)', width=2),
            hoverinfo='none'
        )

        # Node traces
        node_x, node_y, node_z = [], [], []
        node_text = []
        node_color = []
        node_size = []

        for node in sub_g.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            d = sub_g.nodes[node]
            score = d.get('kingpin_score', 20)
            node_size.append(max(6, int(score / 4)))
            node_color.append(score)
            
            node_text.append(
                f"<b>{d['name']} ({node})</b><br>"
                f"Gang: {d['gang']}<br>"
                f"Role: {d['role']}<br>"
                f"Risk: {d['risk_level']}<br>"
                f"Kingpin Score: {score}/100<br>"
                f"City: {d['city']}"
            )

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            text=[sub_g.nodes[n]['name'] for n in sub_g.nodes()],
            textposition="top center",
            textfont=dict(size=9, color="#E2E8F0"),
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Inferno',
                colorbar=dict(title="Kingpin Score", thickness=15),
                line=dict(color='#FFFFFF', width=1)
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="3D Criminal Syndicate Topology & Influence Spheres",
            showlegend=False,
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title=''),
                bgcolor="#0B0F19"
            ),
            paper_bgcolor="#0B0F19",
            font=dict(color="#FFFFFF"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig
