import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (roc_curve, auc, classification_report,
                             confusion_matrix, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

BG=     '#0D1117'; PANEL=  '#161B22'; BORDER= '#30363D'; TEXT=  '#E6EDF3'
SUBTEXT='#8B949E'; C_BLUE= '#58A6FF'; C_GREEN='#3FB950'; C_RED= '#F85149'
C_ORANGE='#E3B341';C_PURPLE='#BC8CFF';C_TEAL= '#39C5CF'; C_PINK='#FF7B72'
C_YELLOW='#F0E68C'

ATTACK_COLORS = {
    'BRUTE_FORCE':C_ORANGE,'CREDENTIAL_STUFFING':C_YELLOW,
    'LATERAL_MOVEMENT':C_PURPLE,'DATA_EXFILTRATION':C_RED,
    'PRIVILEGE_ESCALATION':C_PINK,'None':C_BLUE,
}

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
    'axes.titleweight':'bold','figure.facecolor':BG,'axes.facecolor':PANEL,
    'axes.edgecolor':BORDER,'axes.labelcolor':SUBTEXT,'xtick.color':SUBTEXT,
    'ytick.color':SUBTEXT,'text.color':TEXT,'grid.color':BORDER,
    'grid.linewidth':0.5,'legend.facecolor':PANEL,'legend.edgecolor':BORDER,
    'legend.fontsize':8,
})

def panel(ax, title, subtitle=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(1.2)
    full = title + (f'\n{subtitle}' if subtitle else '')
    ax.set_title(full, color=TEXT, pad=10, fontsize=10.5, fontweight='bold', loc='left')
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    return ax

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('../data/security_logs.csv', parse_dates=['timestamp'])
df['date']    = pd.to_datetime(df['date'])
df['hour']    = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.day_name()
df['week']    = df['timestamp'].dt.isocalendar().week.astype(int)
df['month']   = df['timestamp'].dt.month

for col in ['platform','event_type','status','weekday']:
    le = LabelEncoder()
    df[col+'_enc'] = le.fit_transform(df[col])

FEATURES = ['response_time_ms','bytes_sent','login_attempts','is_vpn',
            'hour','platform_enc','event_type_enc','status_enc','weekday_enc']
df['is_attack'] = (df['label']=='ATTACK').astype(int)
X = df[FEATURES]; y = df['is_attack']
scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.3,
                                           random_state=42, stratify=y)
print(f"  Train:{len(X_tr):,} | Test:{len(X_te):,} | Attack rate:{y.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DEEP DIVE PER ATTACK TYPE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Deep Dive per Attack Type...")
attack_df = df[df['label']=='ATTACK'].copy()
atypes = ['BRUTE_FORCE','CREDENTIAL_STUFFING','LATERAL_MOVEMENT',
          'DATA_EXFILTRATION','PRIVILEGE_ESCALATION']

fig1 = plt.figure(figsize=(22, 20), facecolor=BG)
fig1.suptitle('🔬  DEEP DIVE — Attack Type Analysis\nBehavioral Fingerprinting per Attack Vector',
              fontsize=15, fontweight='bold', color=TEXT, y=0.98, linespacing=1.8)
gs1 = gridspec.GridSpec(3, 3, fig1, hspace=0.55, wspace=0.4,
                        left=0.07, right=0.97, top=0.92, bottom=0.06)

# 1A — Box Response Time
ax = panel(fig1.add_subplot(gs1[0,:2]),'⏱  Response Time Distribution','per Attack Type vs Normal')
ax.grid(True, alpha=0.3, axis='y')
bdata  = [df[df['label']=='NORMAL']['response_time_ms'].values] + \
         [attack_df[attack_df['attack_type']==a]['response_time_ms'].values for a in atypes]
bcolors= [C_BLUE]+[ATTACK_COLORS[a] for a in atypes]
bp = ax.boxplot(bdata, patch_artist=True, widths=0.5, showfliers=True,
                flierprops=dict(marker='o',markersize=2,alpha=0.4),
                medianprops=dict(color='white',linewidth=2))
for patch,c in zip(bp['boxes'],bcolors): patch.set_facecolor(c); patch.set_alpha(0.75)
for w in bp['whiskers']: w.set_color(BORDER); w.set_linewidth(1)
for c in bp['caps']: c.set_color(BORDER)
for fl,c in zip(bp['fliers'],bcolors): fl.set_markerfacecolor(c); fl.set_markeredgecolor(c)
ax.set_xticks(range(len(bdata)))
ax.set_xticklabels(['Normal']+[a.replace('_','\n') for a in atypes], fontsize=8)
ax.set_ylabel('Response Time (ms)', color=SUBTEXT)

# 1B — Bytes per attack type
ax = panel(fig1.add_subplot(gs1[0,2]),'📦  Bytes Sent','by Attack Type (log scale)')
ax.grid(True, alpha=0.3, axis='x')
for i,at in enumerate(atypes):
    v = np.log1p(attack_df[attack_df['attack_type']==at]['bytes_sent'].values)
    ax.barh(i, np.median(v), color=ATTACK_COLORS[at], alpha=0.8, height=0.6, edgecolor=BG)
    ax.errorbar(np.median(v), i, xerr=[[np.median(v)-np.percentile(v,25)],
                                        [np.percentile(v,75)-np.median(v)]],
                fmt='none', color='white', linewidth=2, capsize=4)
ax.set_yticks(range(len(atypes)))
ax.set_yticklabels([a.replace('_','\n') for a in atypes], fontsize=7.5)
ax.set_xlabel('log(Bytes Sent + 1)', color=SUBTEXT)

# 1C — Heatmap hour × attack
ax = panel(fig1.add_subplot(gs1[1,:]),'🕐  Attack Frequency — Hour × Attack Type','Darker = More Events')
ax.grid(False)
heat = attack_df.groupby(['attack_type','hour']).size().unstack(fill_value=0).reindex(atypes)
cmap1 = LinearSegmentedColormap.from_list('atk',[PANEL,'#1f2d4a','#2e4a7a','#3d6bc5',C_BLUE,'#ffffff'])
sns.heatmap(heat, ax=ax, cmap=cmap1, linewidths=0.3, linecolor=BG,
            annot=True, fmt='d', annot_kws={'size':7.5,'color':TEXT},
            cbar_kws={'shrink':0.6,'label':'Count'})
ax.set_xlabel('Hour of Day', color=SUBTEXT); ax.set_ylabel('')
ax.tick_params(labelsize=8)
ax.set_yticklabels([a.replace('_',' ') for a in heat.index], rotation=0, fontsize=8.5)
ax.collections[0].colorbar.ax.yaxis.set_tick_params(color=SUBTEXT)
plt.setp(ax.collections[0].colorbar.ax.yaxis.get_ticklabels(), color=SUBTEXT, fontsize=8)

# 1D — Platform preference
ax = panel(fig1.add_subplot(gs1[2,0]),'☁️  Platform Preference','per Attack Type')
ax.grid(True, alpha=0.3, axis='x')
pp = attack_df.groupby(['attack_type','platform']).size().unstack(fill_value=0).reindex(atypes)
pp_pct = pp.div(pp.sum(axis=1), axis=0) * 100
ax.barh(range(len(atypes)), pp_pct.get('AWS',0), color=C_ORANGE, alpha=0.85, height=0.55, label='AWS')
ax.barh(range(len(atypes)), pp_pct.get('Azure',0), left=pp_pct.get('AWS',0),
        color=C_BLUE, alpha=0.85, height=0.55, label='Azure')
ax.set_yticks(range(len(atypes)))
ax.set_yticklabels([a.replace('_','\n') for a in atypes], fontsize=7.5)
ax.set_xlabel('Share (%)', color=SUBTEXT); ax.legend(loc='lower right'); ax.set_xlim(0,100)

# 1E — VPN usage
ax = panel(fig1.add_subplot(gs1[2,1]),'🔒  VPN Usage Rate','Normal vs Each Attack')
ax.grid(True, alpha=0.3, axis='y')
vpn = {'Normal': df[df['label']=='NORMAL']['is_vpn'].mean()*100}
vpn.update({a: attack_df[attack_df['attack_type']==a]['is_vpn'].mean()*100 for a in atypes})
vc = [C_BLUE]+[ATTACK_COLORS[a] for a in atypes]
bars = ax.bar(range(len(vpn)), list(vpn.values()), color=vc, alpha=0.85, width=0.6, edgecolor=BG)
for bar,v in zip(bars, vpn.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f'{v:.0f}%', ha='center', va='bottom', color=TEXT, fontsize=8, fontweight='bold')
ax.set_xticks(range(len(vpn)))
ax.set_xticklabels([k.replace('_','\n') for k in vpn], fontsize=7.5)
ax.set_ylabel('VPN Usage Rate (%)', color=SUBTEXT)

# 1F — Kill chain
ax = panel(fig1.add_subplot(gs1[2,2]),'⛓  Cyber Kill Chain Mapping','')
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off'); ax.grid(False)
kc = [('Reconnaissance','CREDENTIAL_STUFFING',C_YELLOW,8.5),
      ('Initial Access','BRUTE_FORCE',C_ORANGE,7.0),
      ('Lateral Movement','LATERAL_MOVEMENT',C_PURPLE,5.5),
      ('Privilege Escalation','PRIVILEGE_ESCALATION',C_PINK,4.0),
      ('Exfiltration','DATA_EXFILTRATION',C_RED,2.5)]
for stage,at,color,yp in kc:
    cnt = len(attack_df[attack_df['attack_type']==at])
    ax.barh(yp, cnt/3, left=4.5, height=0.9, color=color, alpha=0.85, edgecolor=BG)
    ax.text(4.3, yp, stage, ha='right', va='center', color=TEXT, fontsize=8, fontweight='bold')
    ax.text(4.5+cnt/3+0.1, yp, f'{cnt}', ha='left', va='center', color=color, fontsize=9, fontweight='bold')
ax.text(5, 9.5,'Kill Chain Stage → Event Count', ha='center', color=SUBTEXT, fontsize=8)

plt.savefig('/home/romayana/SAP/output/deep_dive_attacks.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✔ Done")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — USER BEHAVIOR PROFILING
# ══════════════════════════════════════════════════════════════════════════════
print("[2/4] User Behavior Profiling...")
user_stats = df.groupby('user').agg(
    total_events=('label','count'), attack_events=('is_attack','sum'),
    unique_ips=('source_ip','nunique'), avg_bytes=('bytes_sent','mean'),
    avg_login_attempts=('login_attempts','mean'),
    failed_logins=('status', lambda x:(x=='FAILED').sum()),
    vpn_usage=('is_vpn','mean'), avg_response_time=('response_time_ms','mean'),
).reset_index()
user_stats['attack_rate'] = user_stats['attack_events'] / user_stats['total_events']
user_stats['fail_rate']   = user_stats['failed_logins']  / user_stats['total_events']

rf_list = ['attack_rate','avg_login_attempts','fail_rate','unique_ips','avg_bytes','vpn_usage']
rn = StandardScaler().fit_transform(user_stats[rf_list].fillna(0))
user_stats['risk_score'] = rn.mean(axis=1)
mn,mx = user_stats['risk_score'].min(), user_stats['risk_score'].max()
user_stats['risk_pct']   = ((user_stats['risk_score']-mn)/(mx-mn)*100).round(1)
user_stats['risk_level'] = pd.cut(user_stats['risk_pct'], bins=[-1,33,66,101],
                                   labels=['LOW','MEDIUM','HIGH'])
print(f"  Users: {len(user_stats)} | HIGH:{(user_stats['risk_level']=='HIGH').sum()} "
      f"MED:{(user_stats['risk_level']=='MEDIUM').sum()} "
      f"LOW:{(user_stats['risk_level']=='LOW').sum()}")

fig2 = plt.figure(figsize=(22,20), facecolor=BG)
fig2.suptitle('👤  USER BEHAVIOR PROFILING\nRisk Scoring & Anomalous User Identification',
              fontsize=15, fontweight='bold', color=TEXT, y=0.98, linespacing=1.8)
gs2 = gridspec.GridSpec(3,3,fig2, hspace=0.55, wspace=0.4,
                        left=0.07, right=0.97, top=0.92, bottom=0.06)

# 2A — Risk distribution
ax = panel(fig2.add_subplot(gs2[0,:2]),'🎯  User Risk Score Distribution','Composite score (0–100)')
ax.grid(True, alpha=0.3, axis='y')
for lvl,c in [('LOW',C_GREEN),('MEDIUM',C_ORANGE),('HIGH',C_RED)]:
    ax.hist(user_stats[user_stats['risk_level']==lvl]['risk_pct'],
            bins=30, color=c, alpha=0.75, label=f'{lvl} Risk')
ax.axvline(33, color='white', linewidth=1.5, linestyle='--', alpha=0.5)
ax.axvline(66, color='white', linewidth=1.5, linestyle='--', alpha=0.5)
ax.set_xlabel('Risk Score (0–100)', color=SUBTEXT); ax.set_ylabel('Users', color=SUBTEXT); ax.legend()

# 2B — Pie
ax = panel(fig2.add_subplot(gs2[0,2]),'🔴  Risk Level Breakdown','')
ax.grid(False)
rc = user_stats['risk_level'].value_counts()
wedges,_,auts = ax.pie(
    [rc.get('HIGH',0),rc.get('MEDIUM',0),rc.get('LOW',0)],
    labels=['HIGH','MEDIUM','LOW'], colors=[C_RED,C_ORANGE,C_GREEN],
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.55,edgecolor=BG,linewidth=2),
    textprops={'color':TEXT,'fontsize':9})
for a in auts: a.set_color(BG); a.set_fontweight('bold'); a.set_fontsize(9)

# 2C — Top 15 risky users
ax = panel(fig2.add_subplot(gs2[1,:]),'🚨  Top 15 High-Risk Users','Sorted by risk score')
ax.grid(True, alpha=0.3, axis='x')
top15 = user_stats.nlargest(15,'risk_pct').sort_values('risk_pct')
colors15 = top15['risk_pct'].apply(lambda v: C_RED if v>66 else (C_ORANGE if v>33 else C_GREEN))
bars = ax.barh(range(len(top15)), top15['risk_pct'], color=colors15, height=0.65, edgecolor=BG)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['user'], fontsize=8, fontfamily='monospace')
ax.set_xlabel('Risk Score (0–100)', color=SUBTEXT); ax.set_xlim(0,115)
for i,(_,row) in enumerate(top15.iterrows()):
    ax.text(row['risk_pct']+1, i,
            f"  Score:{row['risk_pct']:.0f}  Attacks:{row['attack_events']:.0f}  "
            f"IPs:{row['unique_ips']:.0f}  FailRate:{row['fail_rate']*100:.0f}%",
            va='center', color=SUBTEXT, fontsize=7.5)

# 2D — Scatter risk vs IPs
ax = panel(fig2.add_subplot(gs2[2,0]),'🌐  Risk Score vs Unique IPs Used','')
ax.grid(True, alpha=0.3)
rc_colors = user_stats['risk_level'].map({'LOW':C_GREEN,'MEDIUM':C_ORANGE,'HIGH':C_RED})
ax.scatter(user_stats['unique_ips'], user_stats['risk_pct'],
           c=rc_colors, alpha=0.6, s=30, edgecolors='none')
ax.set_xlabel('Unique Source IPs', color=SUBTEXT); ax.set_ylabel('Risk Score', color=SUBTEXT)
for lbl,c in [('LOW',C_GREEN),('MEDIUM',C_ORANGE),('HIGH',C_RED)]:
    ax.scatter([],[],color=c,label=lbl,s=30)
ax.legend()

# 2E — Bubble scatter
ax = panel(fig2.add_subplot(gs2[2,1]),'⚡  Bytes vs Login Attempts','Bubble = total events')
ax.grid(True, alpha=0.3)
samp = user_stats.sample(min(150,len(user_stats)), random_state=42)
sc = ax.scatter(samp['avg_login_attempts'], np.log1p(samp['avg_bytes']),
                s=samp['total_events']*2, c=samp['risk_pct'],
                cmap='RdYlGn_r', alpha=0.7, edgecolors=BORDER, linewidths=0.5, vmin=0, vmax=100)
plt.colorbar(sc, ax=ax, label='Risk Score', shrink=0.8)
ax.set_xlabel('Avg Login Attempts', color=SUBTEXT); ax.set_ylabel('log(Avg Bytes+1)', color=SUBTEXT)

# 2F — Activity hours
ax = panel(fig2.add_subplot(gs2[2,2]),'🕐  Activity Hours','High-Risk vs Normal Users')
ax.grid(True, alpha=0.3)
hr_u = user_stats[user_stats['risk_level']=='HIGH']['user'].values
lr_u = user_stats[user_stats['risk_level']=='LOW']['user'].values
hr_h = df[df['user'].isin(hr_u)]['hour'].value_counts().sort_index()
lr_h = df[df['user'].isin(lr_u)]['hour'].value_counts().sort_index()
hrs  = range(24)
hv   = np.array([hr_h.get(h,0) for h in hrs]); hv = hv/hv.max()*100
lv   = np.array([lr_h.get(h,0) for h in hrs]); lv = lv/lv.max()*100
ax.fill_between(hrs, lv, alpha=0.3, color=C_GREEN); ax.plot(hrs, lv, color=C_GREEN, lw=1.5, label='Low-Risk')
ax.fill_between(hrs, hv, alpha=0.5, color=C_RED);   ax.plot(hrs, hv, color=C_RED,   lw=1.5, label='High-Risk')
ax.axvspan(0,6, alpha=0.08, color=C_PURPLE, label='Night Window')
ax.axvspan(22,24, alpha=0.08, color=C_PURPLE)
ax.set_xlabel('Hour of Day', color=SUBTEXT); ax.set_ylabel('Relative Activity (%)', color=SUBTEXT)
ax.set_xticks(range(0,24,3)); ax.legend(fontsize=7.5)

plt.savefig('/home/romayana/SAP/output/user_behavior_profiling.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✔ Done")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — TIME SERIES FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
print("[3/4] Time Series Forecasting...")
daily_a = df[df['label']=='ATTACK'].groupby('date').size().reset_index(name='attacks')
daily_n = df[df['label']=='NORMAL'].groupby('date').size().reset_index(name='normal')
daily   = pd.merge(daily_a, daily_n, on='date', how='outer').fillna(0).sort_values('date').reset_index(drop=True)
daily['total']       = daily['attacks'] + daily['normal']
daily['attack_rate'] = daily['attacks'] / daily['total']
daily['day_idx']     = range(len(daily))
daily['roll7_atk']   = daily['attacks'].rolling(7,min_periods=1).mean()
daily['roll7_rate']  = daily['attack_rate'].rolling(7,min_periods=1).mean()

N_H  = len(daily); N_F = 14
poly = np.poly1d(np.polyfit(daily['day_idx'], daily['attacks'], 3))
tx   = np.linspace(0, N_H+N_F-1, N_H+N_F)
ty   = poly(tx).clip(0)
wpat = daily.groupby(daily['date'].dt.dayofweek)['attacks'].mean().values
seas = np.tile(wpat, int(np.ceil((N_H+N_F)/7)))[:N_H+N_F]
fy   = (ty[N_H:] + (seas[N_H:]-seas[N_H:].mean())*0.5).clip(0)
fstd = fy.std()*0.5
fx   = pd.date_range(daily['date'].max()+pd.Timedelta(days=1), periods=N_F)

baseline   = daily['roll7_rate'].mean()
sigma_rate = daily['roll7_rate'].std()
alert_thresh = baseline + 2*sigma_rate

fig3 = plt.figure(figsize=(22,18), facecolor=BG)
fig3.suptitle('📈  TIME SERIES FORECASTING\nAttack Volume Prediction & Trend Analysis',
              fontsize=15, fontweight='bold', color=TEXT, y=0.98, linespacing=1.8)
gs3 = gridspec.GridSpec(3,2,fig3, hspace=0.55, wspace=0.38,
                        left=0.07, right=0.97, top=0.92, bottom=0.06)

# 3A — Main forecast
ax = panel(fig3.add_subplot(gs3[0,:]),'📅  Daily Attack Volume — Historical + 14-Day Forecast',
           'Polynomial Trend + Weekly Seasonality')
ax.grid(True, alpha=0.3)
ax.fill_between(daily['date'], daily['attacks'], alpha=0.25, color=C_RED)
ax.plot(daily['date'], daily['attacks'], color=C_RED, alpha=0.5, lw=1, label='Daily Attacks')
ax.plot(daily['date'], daily['roll7_atk'], color=C_ORANGE, lw=2, label='7-Day Rolling Avg')
ax.plot(daily['date'], poly(daily['day_idx']).clip(0), color=C_PURPLE, lw=1.5, ls='--', label='Trend (Poly-3)')
ax.fill_between(fx, (fy-fstd*1.96).clip(0), (fy+fstd*1.96).clip(0), alpha=0.25, color=C_BLUE, label='95% CI')
ax.plot(fx, fy, color=C_BLUE, lw=2.5, label='Forecast')
ax.axvline(daily['date'].max(), color='white', lw=1.5, ls=':', alpha=0.7)
ax.text(daily['date'].max(), ax.get_ylim()[1]*0.88, ' Forecast\n Window', color=C_BLUE, fontsize=8)
ax.set_xlabel('Date', color=SUBTEXT); ax.set_ylabel('Attack Count', color=SUBTEXT)
ax.legend(loc='upper left', ncol=3, fontsize=8)

# 3B — Weekly seasonality
ax = panel(fig3.add_subplot(gs3[1,0]),'📆  Weekly Attack Seasonality','Avg attacks per day of week')
ax.grid(True, alpha=0.3, axis='y')
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_a = df[df['label']=='ATTACK'].groupby('weekday').size().reindex(dow_order, fill_value=0)
dow_n = df[df['label']=='NORMAL'].groupby('weekday').size().reindex(dow_order, fill_value=0)
dow_rate = (dow_a/(dow_a+dow_n)*100).round(2)
bar_col = [C_RED if v>dow_rate.mean() else C_BLUE for v in dow_rate]
bars = ax.bar(range(7), dow_rate.values, color=bar_col, alpha=0.85, width=0.6, edgecolor=BG)
ax.axhline(dow_rate.mean(), color=C_ORANGE, lw=1.5, ls='--', label=f'Avg: {dow_rate.mean():.1f}%')
ax.set_xticks(range(7)); ax.set_xticklabels([d[:3] for d in dow_order], fontsize=9)
ax.set_ylabel('Attack Rate (%)', color=SUBTEXT); ax.legend()
for bar,v in zip(bars,dow_rate): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                                          f'{v:.1f}%', ha='center', va='bottom', fontsize=8, color=TEXT)

# 3C — Monthly
ax = panel(fig3.add_subplot(gs3[1,1]),'📊  Monthly Attack Volume Trend','Jan–Mar 2025')
ax.grid(True, alpha=0.3, axis='y')
mon = df.groupby(['month','label']).size().unstack(fill_value=0).reset_index()
w = 0.4
ax.bar([x-w/2 for x in range(len(mon))], mon.get('NORMAL',0), width=w, color=C_BLUE, alpha=0.85, label='Normal', edgecolor=BG)
ax.bar([x+w/2 for x in range(len(mon))], mon.get('ATTACK',0), width=w, color=C_RED,  alpha=0.85, label='Attack', edgecolor=BG)
ax.set_xticks(range(len(mon))); ax.set_xticklabels(['January','February','March'][:len(mon)])
ax.set_ylabel('Event Count', color=SUBTEXT); ax.legend()

# 3D — Rolling rate + alerts
ax = panel(fig3.add_subplot(gs3[2,:]),'📉  Rolling Attack Rate (7-Day) + Threshold Alerts',
           'Red zone = exceeds 2σ above baseline')
ax.grid(True, alpha=0.3)
ax.fill_between(daily['date'], daily['attack_rate'], alpha=0.2, color=C_TEAL)
ax.plot(daily['date'], daily['attack_rate'], color=C_TEAL, alpha=0.5, lw=1, label='Daily Rate')
ax.plot(daily['date'], daily['roll7_rate'], color=C_ORANGE, lw=2, label='7-Day Rolling Rate')
ax.axhline(baseline, color=C_GREEN, lw=1.5, ls='--', label=f'Baseline: {baseline*100:.1f}%')
ax.axhline(alert_thresh, color=C_RED, lw=1.5, ls='--', label=f'Alert (+2σ): {alert_thresh*100:.1f}%')
ax.fill_between(daily['date'], alert_thresh, daily['roll7_rate'],
                where=(daily['roll7_rate']>alert_thresh), alpha=0.35, color=C_RED, label='Alert Zone')
ax.set_xlabel('Date', color=SUBTEXT); ax.set_ylabel('Attack Rate', color=SUBTEXT)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v*100:.1f}%'))
ax.legend(ncol=3, fontsize=8)

plt.savefig('/home/romayana/SAP/output/time_series_forecast.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✔ Done")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — ADVANCED ML
# ══════════════════════════════════════════════════════════════════════════════
print("[4/4] Advanced ML — 3 Models + ROC Analysis...")
models_def = {
    'Random Forest':      RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
    'Logistic Regression':LogisticRegression(max_iter=1000, random_state=42),
}
roc_colors = {'Random Forest':C_GREEN,'Gradient Boosting':C_ORANGE,'Logistic Regression':C_PURPLE}
results = {}

for name, model in models_def.items():
    model.fit(X_tr, y_tr)
    yp   = model.predict_proba(X_te)[:,1]
    ypred= model.predict(X_te)
    fpr, tpr, _ = roc_curve(y_te, yp)
    ra   = auc(fpr, tpr)
    prec,rec,_  = precision_recall_curve(y_te, yp)
    pra  = auc(rec, prec)
    cv   = cross_val_score(model, X_sc, y, cv=5, scoring='f1').mean()
    cm   = confusion_matrix(y_te, ypred)
    rep  = classification_report(y_te, ypred, output_dict=True)
    results[name] = dict(model=model,y_prob=yp,y_pred=ypred,fpr=fpr,tpr=tpr,
                         roc_auc=ra,prec=prec,rec=rec,pr_auc=pra,cv_f1=cv,cm=cm,report=rep)
    print(f"  ✔ {name:<22} AUC={ra:.4f}  F1={rep['1']['f1-score']:.4f}  CV-F1={cv:.4f}")

fig4 = plt.figure(figsize=(22,20), facecolor=BG)
fig4.suptitle('🤖  ADVANCED ML — Multi-Model Comparison\nRandom Forest  |  Gradient Boosting  |  Logistic Regression',
              fontsize=15, fontweight='bold', color=TEXT, y=0.98, linespacing=1.8)
gs4 = gridspec.GridSpec(3,3,fig4, hspace=0.55, wspace=0.4,
                        left=0.07, right=0.97, top=0.92, bottom=0.06)

# 4A — ROC
ax = panel(fig4.add_subplot(gs4[0,0]),'📈  ROC Curves','All Models')
ax.grid(True, alpha=0.3)
for n,r in results.items():
    ax.plot(r['fpr'],r['tpr'],color=roc_colors[n],lw=2,label=f"{n[:15]} ({r['roc_auc']:.3f})")
ax.plot([0,1],[0,1],color=BORDER,lw=1.5,ls='--',label='Random')
ax.fill_between(results['Random Forest']['fpr'],results['Random Forest']['tpr'],alpha=0.07,color=C_GREEN)
ax.set_xlabel('FPR',color=SUBTEXT); ax.set_ylabel('TPR',color=SUBTEXT)
ax.legend(fontsize=7.5,loc='lower right'); ax.set_xlim(0,1); ax.set_ylim(0,1.02)

# 4B — PR Curves
ax = panel(fig4.add_subplot(gs4[0,1]),'📉  Precision–Recall Curves','All Models')
ax.grid(True, alpha=0.3)
for n,r in results.items():
    ax.plot(r['rec'],r['prec'],color=roc_colors[n],lw=2,label=f"{n[:15]} ({r['pr_auc']:.3f})")
ax.axhline(y_te.mean(), color=BORDER, lw=1.5, ls='--', label=f'Baseline ({y_te.mean():.2f})')
ax.set_xlabel('Recall',color=SUBTEXT); ax.set_ylabel('Precision',color=SUBTEXT)
ax.legend(fontsize=7.5); ax.set_xlim(0,1); ax.set_ylim(0,1.02)

# 4C — Metrics comparison
ax = panel(fig4.add_subplot(gs4[0,2]),'📊  Model Performance Comparison','')
ax.grid(True, alpha=0.3, axis='x')
metrics_list = ['Accuracy','Precision','Recall','F1','AUC','CV-F1']
offsets = [-0.22, 0, 0.22]
for idx,(n,r) in enumerate(results.items()):
    rep = r['report']
    vals = [
        (rep['0']['support']*rep['0']['precision']+rep['1']['support']*rep['1']['precision'])/(rep['0']['support']+rep['1']['support']),
        rep['1']['precision'], rep['1']['recall'], rep['1']['f1-score'],
        r['roc_auc'], r['cv_f1'],
    ]
    ypos = [i+offsets[idx] for i in range(len(metrics_list))]
    ax.barh(ypos, vals, height=0.2, color=roc_colors[n], alpha=0.85, edgecolor=BG, label=n)
ax.set_yticks(range(len(metrics_list))); ax.set_yticklabels(metrics_list, fontsize=9)
ax.set_xlabel('Score',color=SUBTEXT); ax.set_xlim(0,1.15)
ax.axvline(1.0,color=BORDER,lw=1,ls='--'); ax.legend(fontsize=7,loc='lower right')

# 4D — Feature Importance
ax = panel(fig4.add_subplot(gs4[1,:2]),'🔑  Feature Importance — Random Forest',
           'Top features driving attack detection')
ax.grid(True, alpha=0.3, axis='x')
rf_m = results['Random Forest']['model']
fi   = pd.Series(rf_m.feature_importances_, index=FEATURES).sort_values()
fi_c = [C_RED if v>fi.mean()+fi.std() else (C_ORANGE if v>fi.mean() else C_BLUE) for v in fi.values]
ax.barh(range(len(fi)), fi.values, color=fi_c, height=0.6, edgecolor=BG)
ax.set_yticks(range(len(fi)))
ax.set_yticklabels([f.replace('_enc','').replace('_',' ').title() for f in fi.index], fontsize=9)
ax.set_xlabel('Feature Importance (Gini)', color=SUBTEXT)
ax.axvline(fi.mean(), color=C_ORANGE, lw=1.5, ls='--', label=f'Mean:{fi.mean():.3f}')
ax.legend()
for i,v in enumerate(fi.values): ax.text(v+0.001, i, f'{v:.3f}', va='center', fontsize=8, color=SUBTEXT)

# 4E — Confusion Matrix (RF)
ax = panel(fig4.add_subplot(gs4[1,2]),'🎯  Confusion Matrix','Random Forest (Test Set)')
ax.grid(False)
cm_rf = results['Random Forest']['cm']
cm_pct= cm_rf/cm_rf.sum()*100
cmap_cm = LinearSegmentedColormap.from_list('cm',[PANEL,'#1b3a5c','#1f6aa5',C_BLUE,'#99d4ff'])
ax.imshow(cm_pct, cmap=cmap_cm, aspect='auto', vmin=0)
for i in range(2):
    for j in range(2):
        lbl = {(0,0):'TN',(0,1):'FP',(1,0):'FN',(1,1):'TP'}[(i,j)]
        ax.text(j,i,f'{lbl}\n{cm_rf[i,j]:,}\n({cm_pct[i,j]:.1f}%)',
                ha='center',va='center',fontsize=11,fontweight='bold',
                color='white' if cm_pct[i,j]<50 else BG)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Pred: Normal','Pred: Attack'],fontsize=9)
ax.set_yticklabels(['True: Normal','True: Attack'],fontsize=9)

# 4F — Threshold analysis
ax = panel(fig4.add_subplot(gs4[2,:]),'⚖️  Decision Threshold Analysis — Random Forest',
           'Precision / Recall / F1 vs Classification Threshold')
ax.grid(True, alpha=0.3)
threshs = np.linspace(0.05,0.95,100)
rf_p = results['Random Forest']['y_prob']
prcs,rcs,f1s,fprs = [],[],[],[]
for t in threshs:
    p_ = (rf_p>=t).astype(int)
    tp_=(( p_==1)&(y_te==1)).sum(); fp_=((p_==1)&(y_te==0)).sum()
    fn_=(( p_==0)&(y_te==1)).sum(); tn_=((p_==0)&(y_te==0)).sum()
    pr_ = tp_/(tp_+fp_+1e-9); rc_ = tp_/(tp_+fn_+1e-9)
    f1_ = 2*pr_*rc_/(pr_+rc_+1e-9); fp_r= fp_/(fp_+tn_+1e-9)
    prcs.append(pr_); rcs.append(rc_); f1s.append(f1_); fprs.append(fp_r)
ax.plot(threshs,prcs,color=C_BLUE,lw=2,label='Precision')
ax.plot(threshs,rcs, color=C_GREEN,lw=2,label='Recall')
ax.plot(threshs,f1s, color=C_ORANGE,lw=2.5,label='F1-Score')
ax.plot(threshs,fprs,color=C_RED,lw=1.5,ls='--',label='FPR')
best_i = np.argmax(f1s); best_t = threshs[best_i]; best_f1 = f1s[best_i]
ax.axvline(best_t, color='white', lw=2, ls=':', label=f'Best Threshold: {best_t:.2f} (F1={best_f1:.3f})')
ax.fill_between(threshs, f1s, alpha=0.1, color=C_ORANGE)
ax.annotate(f'  Optimal\n  T={best_t:.2f}\n  F1={best_f1:.3f}',
            xy=(best_t,best_f1), xytext=(best_t+0.08,best_f1-0.15),
            color=C_ORANGE, fontsize=8.5, fontweight='bold',
            arrowprops=dict(arrowstyle='->',color=C_ORANGE,lw=1.5))
ax.set_xlabel('Classification Threshold',color=SUBTEXT); ax.set_ylabel('Score',color=SUBTEXT)
ax.set_xlim(0.05,0.95); ax.set_ylim(0,1.05); ax.legend(ncol=3)

plt.savefig('/home/romayana/SAP/output/advanced_ml.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✔ Done")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
best_m = max(results, key=lambda k: results[k]['roc_auc'])
print("\n" + "="*62)
print("  FINAL ADVANCED ANALYSIS REPORT")
print("="*62)
print(f"\n  🏆 Best Model : {best_m}")
print(f"     ROC-AUC    : {results[best_m]['roc_auc']:.4f}")
print(f"     F1-Score   : {results[best_m]['report']['1']['f1-score']:.4f}")
print(f"     CV-F1 (5x) : {results[best_m]['cv_f1']:.4f}")
print(f"     Optimal T  : {best_t:.2f}")
hr = (user_stats['risk_level']=='HIGH').sum()
mr = (user_stats['risk_level']=='MEDIUM').sum()
print(f"\n  👤 User Risk : HIGH={hr} | MEDIUM={mr} | LOW={(user_stats['risk_level']=='LOW').sum()}")
print(f"  📅 Peak Day  : {dow_a.idxmax()} | Baseline Rate: {baseline*100:.2f}%")
print(f"  ⚠  Alert >  : {alert_thresh*100:.2f}% (2σ threshold)")
print(f"  🔮 14-day Forecast Avg: {fy.mean():.1f} attacks/day")
top3 = fi.nlargest(3)
print(f"\n  🔑 Top 3 Attack Indicators:")
for feat,val in top3.items():
    print(f"     • {feat:<25} Gini={val:.4f}")
print(f"\n  ✅ All 4 modules complete!")
print("="*62)
