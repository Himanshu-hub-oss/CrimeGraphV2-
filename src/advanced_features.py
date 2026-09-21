"""CrimeGraph AI v4 advanced case, evidence, finance, alerts and reporting services."""
import os, json, hashlib
from datetime import datetime
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
CASE_DIR = os.path.join(DATA, 'cases')
EVID_DIR = os.path.join(DATA, 'evidence_files')
os.makedirs(CASE_DIR, exist_ok=True); os.makedirs(EVID_DIR, exist_ok=True)
CASES_FILE = os.path.join(DATA, 'cases.json')
EVID_FILE = os.path.join(DATA, 'evidence_registry.json')
FIN_FILE = os.path.join(DATA, 'financial_transactions.csv')

ROLES = {
    'ADMIN': ['*'],
    'INVESTIGATING_OFFICER': ['cases','suspects','alerts','reports','evidence'],
    'CYBER_CELL': ['cases','suspects','alerts','reports','evidence','cdr'],
    'FORENSIC_EXPERT': ['cases','suspects','alerts','reports','evidence','finance'],
    'MAGISTRATE_JUDGE': ['cases','reports','evidence'],
}

def _load_json(path, default):
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: return default

def _save_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)

def load_cases(): return _load_json(CASES_FILE, [])
def save_cases(x): _save_json(CASES_FILE, x)
def load_evidence(): return _load_json(EVID_FILE, [])
def save_evidence(x): _save_json(EVID_FILE, x)

def create_case(case_id, title, crime_type, priority, officer, description=''):
    cases=load_cases()
    if any(c['case_id']==case_id for c in cases): return False, 'Case ID already exists.'
    cases.append({'case_id':case_id,'title':title,'crime_type':crime_type,'priority':priority,'status':'Open','lead_officer':officer,'description':description,'created_at':datetime.now().isoformat(timespec='seconds'),'updated_at':datetime.now().isoformat(timespec='seconds')})
    save_cases(cases); return True, 'Case created.'

def update_case_status(case_id, status):
    cases=load_cases(); found=False
    for c in cases:
        if c['case_id']==case_id: c['status']=status; c['updated_at']=datetime.now().isoformat(timespec='seconds'); found=True
    if found: save_cases(cases)
    return found

def generate_demo_financial_data():
    if os.path.exists(FIN_FILE): return pd.read_csv(FIN_FILE)
    edges=pd.read_csv(os.path.join(DATA,'network_edges.csv'))
    rows=[]
    for i,r in edges.iterrows():
        seed=(i+1)*7919
        amount=round(50000 + (seed % 4950000) + r['interaction_weight']*1375.25,2)
        rows.append({'transaction_id':f'TXN{i+1:05d}','timestamp':f"2026-08-{(i%28)+1:02d} {(i%23):02d}:{(i*7)%60:02d}:00",'source_suspect_id':r['source_suspect_id'],'target_suspect_id':r['target_suspect_id'],'amount_inr':amount,'channel':['Bank Transfer','UPI','Cash Deposit','Wallet'][i%4],'purpose':['Goods','Loan','Consulting','Unknown'][i%4],'risk_flag':'Review' if amount>3500000 else 'Normal'})
    df=pd.DataFrame(rows); df.to_csv(FIN_FILE,index=False); return df

def register_uploaded_evidence(uploaded_file, case_id, officer_badge, role, evidence_type, description=''):
    raw=uploaded_file.getbuffer(); digest=hashlib.sha256(raw).hexdigest(); safe=os.path.basename(uploaded_file.name).replace(' ','_')
    evidence_id='EVID-UP-'+digest[:12].upper(); path=os.path.join(EVID_DIR, evidence_id+'_'+safe)
    with open(path,'wb') as f: f.write(raw)
    reg=load_evidence(); rec={'evidence_id':evidence_id,'case_id':case_id,'filename':safe,'path':path,'size_bytes':len(raw),'sha256':digest,'evidence_type':evidence_type,'description':description,'officer_badge':officer_badge,'role':role,'uploaded_at':datetime.now().isoformat(timespec='seconds')}; reg.append(rec); save_evidence(reg); return rec

def build_alerts(dp, graph_engine, finance_df):
    alerts=[]
    for _,r in dp.suspects_df.iterrows():
        score={'Critical':90,'High':70,'Medium':45,'Low':15}.get(r.get('risk_level'),20)
        node=graph_engine.G.nodes.get(r['suspect_id'],{})
        if node.get('degree',0)>=10: score+=10; alerts.append({'priority':min(score,100),'type':'Network Hub','entity':r['suspect_id'],'message':f"High-connectivity node: degree {node.get('degree',0)}"})
        if r.get('risk_level') in ['Critical','High']: alerts.append({'priority':score,'type':'AI Risk','entity':r['suspect_id'],'message':f"Model risk level: {r.get('risk_level')} — review evidence before action."})
    if not finance_df.empty:
        for _,r in finance_df.nlargest(8,'amount_inr').iterrows():
            if r['amount_inr']>=3500000: alerts.append({'priority':85,'type':'Financial Flow','entity':r['transaction_id'],'message':f"Large transaction ₹{r['amount_inr']:,.2f}: {r['source_suspect_id']} → {r['target_suspect_id']}"})
    cdr=dp.cdr_df
    if not cdr.empty:
        for _,r in cdr.nlargest(8,'duration_seconds').iterrows():
            if r['duration_seconds']>=1200: alerts.append({'priority':75,'type':'CDR Anomaly','entity':r['call_id'],'message':f"Long call {r['duration_seconds']} sec: {r['caller_id']} → {r['callee_id']}"})
    posts=getattr(dp,'posts_df',pd.DataFrame())
    if not posts.empty and 'ai_sentiment_flag' in posts:
        for _,r in posts[posts['ai_sentiment_flag'].astype(str).str.contains('Threat|Suspicious|Coded',case=False,regex=True)].head(10).iterrows():
            alerts.append({'priority':80,'type':'NLP Threat','entity':r['post_id'],'message':f"Flagged social intelligence: {r['ai_sentiment_flag']} ({r['suspect_id']})"})
    return pd.DataFrame(alerts).sort_values('priority',ascending=False).reset_index(drop=True) if alerts else pd.DataFrame(columns=['priority','type','entity','message'])

def investigation_report(dp, graph_engine, suspect_id, finance_df, case_id=''):
    s=dp.suspects_df[dp.suspects_df.suspect_id==suspect_id]
    if s.empty: return None
    r=s.iloc[0]; node=graph_engine.G.nodes.get(suspect_id,{})
    linked=finance_df[(finance_df.source_suspect_id==suspect_id)|(finance_df.target_suspect_id==suspect_id)] if not finance_df.empty else pd.DataFrame()
    return {'case_id':case_id,'suspect_id':suspect_id,'name':r['name'],'risk_level':r['risk_level'],'gang':r['gang_affiliation'],'role':r['role_in_network'],'city':r['city'],'primary_crime':r['primary_crime_type'],'network_degree':node.get('degree',0),'kingpin_score':node.get('kingpin_score',0),'financial_transaction_count':len(linked),'financial_total_inr':float(linked.amount_inr.sum()) if not linked.empty else 0,'generated_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'disclaimer':'AI-generated investigative aid. Not proof of guilt. Validate findings against lawful evidence and human review.'}
