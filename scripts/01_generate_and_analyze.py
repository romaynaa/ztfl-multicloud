import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'figure.facecolor': '#0F1117',
    'axes.facecolor': '#1A1D27',
    'axes.edgecolor': '#2D3250',
    'axes.labelcolor': '#C9D1D9',
    'xtick.color': '#8B949E',
    'ytick.color': '#8B949E',
    'text.color': '#C9D1D9',
    'grid.color': '#2D3250',
    'grid.linewidth': 0.5,
    'legend.facecolor': '#1A1D27',
    'legend.edgecolor': '#2D3250',
})

# ─── COLORS ───────────────────────────────────────────────────────────────────
C_BLUE   = '#58A6FF'
C_GREEN  = '#3FB950'
C_RED    = '#F85149'
C_ORANGE = '#E3B341'
C_PURPLE = '#A5A0FF'
C_TEAL   = '#39D353'
C_PINK   = '#FF7B72'

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  SECURITY LOG ANALYSIS SYSTEM")
print("  Multi-Cloud Environment (AWS + Azure)")
print("=" * 60)
print("\n[1/5] Generating realistic security log dataset...")

N_NORMAL  = 4500
N_ATTACK  = 500
N_TOTAL   = N_NORMAL + N_ATTACK

# Legitimate IP pools (AWS & Azure subnets)
aws_ips   = [f"10.0.{np.random.randint(1,15)}.{np.random.randint(1,254)}" for _ in range(200)]
azure_ips = [f"172.16.{np.random.randint(1,10)}.{np.random.randint(1,254)}" for _ in range(150)]
ext_ips   = [f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(80)]
attack_ips = [f"{np.random.choice([185,45,91,104,141])}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(50)]

all_legit_ips = aws_ips + azure_ips + ext_ips

users_normal  = [f"user_{i:04d}" for i in range(1, 201)]
users_service = [f"svc_{s}" for s in ['api','db','auth','worker','scheduler']]
users_attack  = [f"user_{i:04d}" for i in [9999, 8888, 7777, 6666, 5555]]

base_time = datetime(2025, 1, 1, 0, 0, 0)

# ── Normal traffic ──────────────────────────────────────────────────────────
normal_times = [base_time + timedelta(
    days=np.random.randint(0, 90),
    hours=int(np.random.choice(range(24),
                          p=(lambda a: a/a.sum())(np.array([
                              0.01,0.005,0.003,0.003,0.004,0.008,0.025,0.065,
                              0.085,0.085,0.075,0.065,0.065,0.065,0.065,0.065,
                              0.055,0.055,0.045,0.038,0.030,0.025,0.018,0.012])))),
    minutes=np.random.randint(0, 60),
    seconds=np.random.randint(0, 60)
) for _ in range(N_NORMAL)]

normal_df = pd.DataFrame({
    'timestamp':       normal_times,
    'source_ip':       np.random.choice(all_legit_ips, N_NORMAL),
    'user':            np.random.choice(users_normal + users_service, N_NORMAL,
                                        p=[0.004]*200 + [0.04]*5),
    'platform':        np.random.choice(['AWS', 'Azure'], N_NORMAL, p=[0.55, 0.45]),
    'event_type':      np.random.choice(
                           ['LOGIN_SUCCESS','API_CALL','DATA_ACCESS','CONFIG_READ',
                            'FILE_READ','SESSION_START','LOGOUT'],
                           N_NORMAL, p=[0.20,0.30,0.20,0.10,0.10,0.05,0.05]),
    'status':          np.random.choice(['SUCCESS','FAILED'], N_NORMAL, p=[0.95, 0.05]),
    'response_time_ms':np.random.normal(120, 30, N_NORMAL).clip(20, 400).astype(int),
    'bytes_sent':      np.random.exponential(5000, N_NORMAL).clip(100, 50000).astype(int),
    'login_attempts':  np.random.choice([1,2,3], N_NORMAL, p=[0.90,0.08,0.02]),
    'is_vpn':          np.random.choice([0,1], N_NORMAL, p=[0.85, 0.15]),
    'label':           'NORMAL',
    'attack_type':     'None'
})

# ── Attack traffic ──────────────────────────────────────────────────────────
# Attack time distribution — mostly off-hours (night)
_ap = np.array([0.08,0.08,0.08,0.07,0.06,0.04,0.02,0.01,
               0.01,0.01,0.02,0.02,0.02,0.02,0.02,0.03,
               0.03,0.04,0.05,0.06,0.07,0.07,0.07,0.07])
attack_hours = np.random.choice(range(24), N_ATTACK, p=_ap/_ap.sum())
attack_times = [base_time + timedelta(
    days=np.random.randint(0, 90),
    hours=int(attack_hours[i]),
    minutes=np.random.randint(0, 60),
    seconds=np.random.randint(0, 60)
) for i in range(N_ATTACK)]

attack_types_list = np.random.choice(
    ['BRUTE_FORCE','CREDENTIAL_STUFFING','LATERAL_MOVEMENT',
     'DATA_EXFILTRATION','PRIVILEGE_ESCALATION'],
    N_ATTACK, p=[0.30, 0.25, 0.20, 0.15, 0.10]
)

def attack_event(atype):
    mapping = {
        'BRUTE_FORCE':          ('LOGIN_FAILED',  'FAILED', np.random.randint(10,50), np.random.randint(50,200),   np.random.randint(50, 5000)),
        'CREDENTIAL_STUFFING':  ('LOGIN_FAILED',  'FAILED', np.random.randint(5,20),  np.random.randint(80,200),   np.random.randint(100,3000)),
        'LATERAL_MOVEMENT':     ('API_CALL',      'SUCCESS',np.random.randint(1,3),   np.random.randint(50,150),   np.random.randint(500,5000)),
        'DATA_EXFILTRATION':    ('DATA_ACCESS',   'SUCCESS',np.random.randint(1,2),   np.random.randint(2000,8000),np.random.randint(100000,5000000)),
        'PRIVILEGE_ESCALATION': ('CONFIG_CHANGE', 'SUCCESS',np.random.randint(1,3),   np.random.randint(100,300),  np.random.randint(1000,10000)),
    }
    return mapping[atype]

attack_events = [attack_event(a) for a in attack_types_list]
ev_type, ev_status, ev_attempts, ev_rt, ev_bytes = zip(*attack_events)

attack_df = pd.DataFrame({
    'timestamp':       attack_times,
    'source_ip':       np.random.choice(attack_ips, N_ATTACK),
    'user':            np.random.choice(users_attack + users_normal[:20], N_ATTACK),
    'platform':        np.random.choice(['AWS', 'Azure'], N_ATTACK, p=[0.6, 0.4]),
    'event_type':      list(ev_type),
    'status':          list(ev_status),
    'response_time_ms':list(ev_rt),
    'bytes_sent':      list(ev_bytes),
    'login_attempts':  list(ev_attempts),
    'is_vpn':          np.random.choice([0,1], N_ATTACK, p=[0.40, 0.60]),
    'label':           'ATTACK',
    'attack_type':     attack_types_list
})

# ── Combine & clean ─────────────────────────────────────────────────────────
df = pd.concat([normal_df, attack_df], ignore_index=True)
df = df.sort_values('timestamp').reset_index(drop=True)
df['hour']    = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.day_name()
df['date']    = df['timestamp'].dt.date
df['week']    = df['timestamp'].dt.isocalendar().week.astype(int)

df.to_csv('../data/security_logs.csv', index=False)
print(f"  ✔ Dataset created: {len(df):,} records | {N_ATTACK} attacks ({N_ATTACK/N_TOTAL*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Running exploratory statistics...")

total = len(df)
attacks = (df['label']=='ATTACK').sum()
normal  = (df['label']=='NORMAL').sum()

print(f"\n  {'METRIC':<35} {'VALUE':>10}")
print(f"  {'-'*46}")
print(f"  {'Total Log Entries':<35} {total:>10,}")
print(f"  {'Normal Events':<35} {normal:>10,}")
print(f"  {'Attack Events':<35} {attacks:>10,}")
print(f"  {'Attack Rate':<35} {attacks/total*100:>9.2f}%")
print(f"  {'Unique Users':<35} {df['user'].nunique():>10,}")
print(f"  {'Unique Source IPs':<35} {df['source_ip'].nunique():>10,}")
print(f"  {'Date Range':<35} {str(df['date'].min())+' → '+str(df['date'].max()):>10}")
print(f"  {'Avg Response Time (Normal)':<35} {df[df['label']=='NORMAL']['response_time_ms'].mean():>9.1f} ms")
print(f"  {'Avg Response Time (Attack)':<35} {df[df['label']=='ATTACK']['response_time_ms'].mean():>9.1f} ms")
print(f"  {'Avg Bytes Sent (Normal)':<35} {df[df['label']=='NORMAL']['bytes_sent'].mean():>9,.0f}")
print(f"  {'Avg Bytes Sent (Attack)':<35} {df[df['label']=='ATTACK']['bytes_sent'].mean():>9,.0f}")

# Top attack types
atk_dist = df[df['label']=='ATTACK']['attack_type'].value_counts()
print(f"\n  Attack Type Distribution:")
for atype, cnt in atk_dist.items():
    print(f"    • {atype:<25} {cnt:>4} ({cnt/attacks*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ANOMALY DETECTION — Isolation Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Running Isolation Forest anomaly detection...")

le_platform   = LabelEncoder()
le_event      = LabelEncoder()
le_status     = LabelEncoder()
le_weekday    = LabelEncoder()

df_ml = df.copy()
df_ml['platform_enc'] = le_platform.fit_transform(df_ml['platform'])
df_ml['event_enc']    = le_event.fit_transform(df_ml['event_type'])
df_ml['status_enc']   = le_status.fit_transform(df_ml['status'])
df_ml['weekday_enc']  = le_weekday.fit_transform(df_ml['weekday'])

features = ['response_time_ms','bytes_sent','login_attempts','is_vpn',
            'hour','platform_enc','event_enc','status_enc','weekday_enc']

scaler = StandardScaler()
X = scaler.fit_transform(df_ml[features])

iso = IsolationForest(n_estimators=200, contamination=0.10, random_state=42, n_jobs=-1)
df['anomaly_score'] = iso.fit(X).score_samples(X)
df['anomaly_pred']  = iso.predict(X)  # -1 = anomaly, 1 = normal
df['is_anomaly']    = (df['anomaly_pred'] == -1).astype(int)

# Confusion metrics
tp = ((df['is_anomaly']==1) & (df['label']=='ATTACK')).sum()
fp = ((df['is_anomaly']==1) & (df['label']=='NORMAL')).sum()
tn = ((df['is_anomaly']==0) & (df['label']=='NORMAL')).sum()
fn = ((df['is_anomaly']==0) & (df['label']=='ATTACK')).sum()

precision = tp / (tp + fp) if (tp+fp) > 0 else 0
recall    = tp / (tp + fn) if (tp+fn) > 0 else 0
f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
accuracy  = (tp+tn)/total

print(f"  ✔ Model: Isolation Forest (contamination=10%, n_estimators=200)")
print(f"  ✔ Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Accuracy: {accuracy:.3f}")
print(f"  ✔ TP={tp} | FP={fp} | TN={tn} | FN={fn}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING — KMeans
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Running KMeans clustering on attack patterns...")

attack_data = df_ml[df_ml['label']=='ATTACK'][features].copy()
X_atk = scaler.transform(attack_data)

km = KMeans(n_clusters=5, random_state=42, n_init=10)
df.loc[df['label']=='ATTACK', 'cluster'] = km.fit_predict(X_atk)
df['cluster'] = df['cluster'].fillna(-1).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Generating security dashboard...")

fig = plt.figure(figsize=(22, 28), facecolor='#0F1117')
fig.suptitle('🔐  SECURITY LOG ANALYSIS DASHBOARD\n'
             'Multi-Cloud Environment  |  AWS + Azure  |  Jan–Mar 2025',
             fontsize=16, fontweight='bold', color='#E6EDF3',
             y=0.98, linespacing=1.8)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.52, wspace=0.38,
                       left=0.07, right=0.96, top=0.93, bottom=0.04)

PANEL_EDGE = '#2D3250'

def styled_ax(ax, title):
    ax.set_facecolor('#1A1D27')
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
        spine.set_linewidth(1.2)
    ax.set_title(title, color='#E6EDF3', pad=10, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return ax

# ── [A] Timeline: Events per Day ────────────────────────────────────────────
ax_a = styled_ax(fig.add_subplot(gs[0, :2]), '📅  Daily Event Volume (Normal vs Attack)')

daily = df.groupby(['date','label']).size().unstack(fill_value=0).reset_index()
daily['date'] = pd.to_datetime(daily['date'])
ax_a.fill_between(daily['date'], daily.get('NORMAL',0),
                  alpha=0.25, color=C_BLUE, label='Normal')
ax_a.plot(daily['date'], daily.get('NORMAL',0),
          color=C_BLUE, linewidth=1.5)
ax_a.fill_between(daily['date'], daily.get('ATTACK',0),
                  alpha=0.4, color=C_RED, label='Attack')
ax_a.plot(daily['date'], daily.get('ATTACK',0),
          color=C_RED, linewidth=1.5)
ax_a.set_xlabel('Date', color='#8B949E')
ax_a.set_ylabel('Event Count', color='#8B949E')
ax_a.legend(facecolor='#1A1D27', edgecolor=PANEL_EDGE, labelcolor='#C9D1D9')
ax_a.tick_params(colors='#8B949E')

# ── [B] Attack Type Donut ────────────────────────────────────────────────────
ax_b = styled_ax(fig.add_subplot(gs[0, 2]), '🎯  Attack Type Distribution')
ax_b.grid(False)

atk_vals  = atk_dist.values
atk_names = [n.replace('_', '\n') for n in atk_dist.index]
colors_pie = [C_RED, C_ORANGE, C_PURPLE, C_PINK, C_TEAL]
wedges, texts, autotexts = ax_b.pie(
    atk_vals, labels=atk_names, autopct='%1.1f%%',
    colors=colors_pie[:len(atk_vals)],
    pctdistance=0.8, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='#0F1117', linewidth=2),
    textprops={'color': '#C9D1D9', 'fontsize': 7.5}
)
for at in autotexts:
    at.set_color('#0F1117'); at.set_fontsize(7.5); at.set_fontweight('bold')

# ── [C] Hourly Heatmap ───────────────────────────────────────────────────────
ax_c = styled_ax(fig.add_subplot(gs[1, :2]), '🕐  Attack Heatmap by Hour & Day of Week')
ax_c.grid(False)

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heat_data = df[df['label']=='ATTACK'].groupby(['weekday','hour']).size().unstack(fill_value=0)
heat_data = heat_data.reindex([d for d in day_order if d in heat_data.index])

cmap = LinearSegmentedColormap.from_list('threat',
    ['#1A1D27','#2D1F3D','#5C1F5F','#A0264C','#E3293A','#F85149'])
sns.heatmap(heat_data, ax=ax_c, cmap=cmap, linewidths=0.4, linecolor='#0F1117',
            annot=True, fmt='d', annot_kws={'size':8, 'color':'#E6EDF3'},
            cbar_kws={'shrink':0.8})
ax_c.set_xlabel('Hour of Day', color='#8B949E')
ax_c.set_ylabel('', color='#8B949E')
ax_c.tick_params(colors='#8B949E', labelsize=8)
ax_c.collections[0].colorbar.ax.yaxis.set_tick_params(color='#8B949E')
plt.setp(ax_c.collections[0].colorbar.ax.yaxis.get_ticklabels(), color='#8B949E')

# ── [D] Anomaly Score Distribution ──────────────────────────────────────────
ax_d = styled_ax(fig.add_subplot(gs[1, 2]), '🔍  Anomaly Score Distribution')

normal_scores = df[df['label']=='NORMAL']['anomaly_score']
attack_scores = df[df['label']=='ATTACK']['anomaly_score']
ax_d.hist(normal_scores, bins=50, color=C_BLUE, alpha=0.6, label='Normal', density=True)
ax_d.hist(attack_scores, bins=50, color=C_RED,  alpha=0.7, label='Attack', density=True)
threshold = df['anomaly_score'].quantile(0.10)
ax_d.axvline(threshold, color=C_ORANGE, linewidth=2, linestyle='--', label=f'Threshold')
ax_d.set_xlabel('Anomaly Score', color='#8B949E')
ax_d.set_ylabel('Density', color='#8B949E')
ax_d.legend(facecolor='#1A1D27', edgecolor=PANEL_EDGE, labelcolor='#C9D1D9', fontsize=8)
ax_d.tick_params(colors='#8B949E')

# ── [E] Login Attempts vs Bytes — Scatter ───────────────────────────────────
ax_e = styled_ax(fig.add_subplot(gs[2, :2]),
                 '⚡  Login Attempts vs Bytes Transferred (Colored by Label)')

sample_n = df[df['label']=='NORMAL'].sample(800, random_state=42)
sample_a = df[df['label']=='ATTACK']
ax_e.scatter(sample_n['login_attempts'], np.log1p(sample_n['bytes_sent']),
             c=C_BLUE, alpha=0.3, s=15, label='Normal')
ax_e.scatter(sample_a['login_attempts'], np.log1p(sample_a['bytes_sent']),
             c=C_RED,  alpha=0.6, s=25, marker='X', label='Attack',
             edgecolors='#FF0000', linewidths=0.3)
ax_e.set_xlabel('Login Attempts', color='#8B949E')
ax_e.set_ylabel('log(Bytes Sent + 1)', color='#8B949E')
ax_e.legend(facecolor='#1A1D27', edgecolor=PANEL_EDGE, labelcolor='#C9D1D9')
ax_e.tick_params(colors='#8B949E')

# ── [F] Platform Comparison ──────────────────────────────────────────────────
ax_f = styled_ax(fig.add_subplot(gs[2, 2]), '☁️  Attack Rate by Platform')
ax_f.grid(False)

plat_stats = df.groupby('platform')['label'].apply(
    lambda x: (x=='ATTACK').sum()/len(x)*100).reset_index()
plat_stats.columns = ['platform','attack_rate']
bars = ax_f.bar(plat_stats['platform'], plat_stats['attack_rate'],
                color=[C_ORANGE, C_PURPLE], width=0.5,
                edgecolor='#0F1117', linewidth=1.5)
for bar, val in zip(bars, plat_stats['attack_rate']):
    ax_f.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
              f'{val:.1f}%', ha='center', va='bottom',
              color='#E6EDF3', fontweight='bold', fontsize=11)
ax_f.set_ylabel('Attack Rate (%)', color='#8B949E')
ax_f.tick_params(colors='#8B949E')
ax_f.set_ylim(0, plat_stats['attack_rate'].max() * 1.3)

# ── [G] Confusion Matrix ─────────────────────────────────────────────────────
ax_g = styled_ax(fig.add_subplot(gs[3, 0]), '🎯  Confusion Matrix (Isolation Forest)')
ax_g.grid(False)

cm_data = np.array([[tn, fp], [fn, tp]])
cm_labels= [['TN', 'FP'], ['FN', 'TP']]
im = ax_g.imshow(cm_data, cmap='RdYlGn', aspect='auto', vmin=0)
for i in range(2):
    for j in range(2):
        ax_g.text(j, i, f'{cm_labels[i][j]}\n{cm_data[i][j]:,}',
                  ha='center', va='center', fontsize=12, fontweight='bold',
                  color='#0F1117')
ax_g.set_xticks([0,1]); ax_g.set_yticks([0,1])
ax_g.set_xticklabels(['Pred: Normal','Pred: Attack'], color='#C9D1D9', fontsize=8)
ax_g.set_yticklabels(['True: Normal','True: Attack'], color='#C9D1D9', fontsize=8)

# ── [H] Model Metrics Bar ────────────────────────────────────────────────────
ax_h = styled_ax(fig.add_subplot(gs[3, 1]), '📊  Model Performance Metrics')

metrics      = ['Accuracy','Precision','Recall','F1-Score']
metric_vals  = [accuracy, precision, recall, f1]
metric_colors= [C_GREEN, C_BLUE, C_ORANGE, C_PURPLE]
bars_h = ax_h.barh(metrics, metric_vals, color=metric_colors,
                   height=0.5, edgecolor='#0F1117')
for bar, val in zip(bars_h, metric_vals):
    ax_h.text(val + 0.005, bar.get_y() + bar.get_height()/2,
              f'{val:.3f}', va='center', color='#E6EDF3',
              fontsize=10, fontweight='bold')
ax_h.set_xlim(0, 1.1)
ax_h.set_xlabel('Score', color='#8B949E')
ax_h.tick_params(colors='#8B949E')

# ── [I] Top Suspicious IPs ───────────────────────────────────────────────────
ax_i = styled_ax(fig.add_subplot(gs[3, 2]), '🚨  Top 10 Suspicious Source IPs')

top_ips = (df[df['label']=='ATTACK']['source_ip']
           .value_counts().head(10).sort_values())
bars_i = ax_i.barh(range(len(top_ips)), top_ips.values,
                   color=C_RED, alpha=0.8, height=0.6,
                   edgecolor='#0F1117')
ax_i.set_yticks(range(len(top_ips)))
ax_i.set_yticklabels([ip[:15] for ip in top_ips.index],
                     fontsize=7.5, color='#C9D1D9', fontfamily='monospace')
for bar, val in zip(bars_i, top_ips.values):
    ax_i.text(val + 0.1, bar.get_y() + bar.get_height()/2,
              str(val), va='center', color='#E6EDF3', fontsize=8)
ax_i.set_xlabel('Attack Count', color='#8B949E')
ax_i.tick_params(colors='#8B949E')

plt.savefig('../output/security_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0F1117', edgecolor='none')
plt.close()
print("  ✔ Dashboard saved!")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ANALYSIS SUMMARY REPORT")
print("=" * 60)

# Risk severity per attack type
risk_map = {
    'DATA_EXFILTRATION':    ('CRITICAL', '🔴'),
    'PRIVILEGE_ESCALATION': ('HIGH',     '🟠'),
    'LATERAL_MOVEMENT':     ('HIGH',     '🟠'),
    'BRUTE_FORCE':          ('MEDIUM',   '🟡'),
    'CREDENTIAL_STUFFING':  ('MEDIUM',   '🟡'),
}
print("\n  THREAT INTELLIGENCE:")
for atype, cnt in atk_dist.items():
    sev, icon = risk_map.get(atype, ('LOW','🟢'))
    print(f"  {icon} {atype:<25} {cnt:>4} events  [{sev}]")

# Peak attack hours
peak_hour = df[df['label']=='ATTACK']['hour'].value_counts().idxmax()
peak_day  = df[df['label']=='ATTACK']['weekday'].value_counts().idxmax()
print(f"\n  PEAK ATTACK TIME:  {peak_hour:02d}:00 - {peak_hour+1:02d}:00 | {peak_day}")

# Most targeted platform
top_plat = df[df['label']=='ATTACK']['platform'].value_counts().idxmax()
print(f"  MOST TARGETED:     {top_plat}")

# Anomaly detection result
detected = df[(df['label']=='ATTACK')&(df['is_anomaly']==1)].shape[0]
missed   = df[(df['label']=='ATTACK')&(df['is_anomaly']==0)].shape[0]
print(f"\n  ANOMALY DETECTION RESULT:")
print(f"  ✔ Attacks Detected : {detected} / {attacks} ({detected/attacks*100:.1f}%)")
print(f"  ✘ Attacks Missed   : {missed} / {attacks} ({missed/attacks*100:.1f}%)")
print(f"  ⚠ False Alarms     : {fp} events flagged incorrectly")

print(f"\n  RECOMMENDATIONS:")
print(f"  1. Block IPs from Autonomous Systems: 185.x, 45.x, 91.x, 104.x, 141.x")
print(f"  2. Enforce MFA for logins >3 attempts (detects Brute Force)")
print(f"  3. Alert on bytes_sent > 100,000 during off-peak hours (23:00–05:00)")
print(f"  4. Investigate {top_plat} for potential misconfiguration")
print(f"  5. Deploy Zero-Trust policies for {peak_day} {peak_hour:02d}:00 peak window")

print(f"\n  OUTPUT FILES:")
print(f"  • security_logs.csv       — Raw log dataset (5,000 records)")
print(f"  • security_dashboard.png  — Visual analysis dashboard")
print("\n" + "=" * 60)
print("  ✅ Analysis Complete!")
print("=" * 60)
