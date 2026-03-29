import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── THEME ────────────────────────────────────────────────────────────────────
BG     = '#0D1117'
PANEL  = '#161B22'
BORDER = '#21262D'
TEXT   = '#E6EDF3'
MUTED  = '#8B949E'

C_RED   = '#FF4444'
C_ORG   = '#FFA500'
C_YEL   = '#E3B341'
C_GRN   = '#3FB950'
C_BLUE  = '#58A6FF'
C_PUR   = '#BC8CFF'
C_PINK  = '#FF7B72'
C_CYAN  = '#79C0FF'
C_TEAL  = '#2EA043'

ATK_PAL = {
    'DATA_EXFILTRATION':    '#FF4444',
    'PRIVILEGE_ESCALATION': '#FF7B72',
    'LATERAL_MOVEMENT':     '#BC8CFF',
    'BRUTE_FORCE':          '#FFA500',
    'CREDENTIAL_STUFFING':  '#E3B341',
}

RISK_PAL = {
    'CRITICAL': '#FF4444',
    'HIGH':     '#FFA500',
    'MEDIUM':   '#E3B341',
    'LOW':      '#58A6FF',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'figure.facecolor': BG, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORDER, 'axes.labelcolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': TEXT, 'grid.color': BORDER,
    'legend.facecolor': PANEL, 'legend.edgecolor': BORDER,
})

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & ENRICH
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv('../data/security_logs.csv', parse_dates=['timestamp'])
atk = df[df['label'] == 'ATTACK'].copy()

print("="*65)
print("  NETWORK GRAPH ANALYSIS  —  IP ↔ User ↔ Platform")
print("="*65)

# ── Node stats ────────────────────────────────────────────────────────────────
# IP nodes
ip_stats = atk.groupby('source_ip').agg(
    attack_count   = ('label',       'count'),
    unique_users   = ('user',        'nunique'),
    unique_platforms=('platform',    'nunique'),
    dominant_attack= ('attack_type', lambda x: x.value_counts().index[0]),
    bytes_total    = ('bytes_sent',  'sum'),
    avg_attempts   = ('login_attempts','mean'),
    vpn_ratio      = ('is_vpn',      'mean'),
).reset_index()
ip_stats['threat_score'] = (
    (ip_stats['attack_count']  / ip_stats['attack_count'].max())  * 35 +
    (ip_stats['unique_users']  / ip_stats['unique_users'].max())  * 25 +
    (ip_stats['bytes_total']   / ip_stats['bytes_total'].max())   * 25 +
    ip_stats['vpn_ratio']                                          * 15
).clip(0, 100)

def risk(s):
    if s >= 65: return 'CRITICAL'
    if s >= 40: return 'HIGH'
    if s >= 20: return 'MEDIUM'
    return 'LOW'
ip_stats['risk'] = ip_stats['threat_score'].apply(risk)

# User nodes
user_stats = atk.groupby('user').agg(
    attack_count    = ('label',        'count'),
    unique_ips      = ('source_ip',    'nunique'),
    unique_platforms= ('platform',     'nunique'),
    dominant_attack = ('attack_type',  lambda x: x.value_counts().index[0]),
    bytes_total     = ('bytes_sent',   'sum'),
    avg_attempts    = ('login_attempts','mean'),
    vpn_ratio       = ('is_vpn',       'mean'),
).reset_index()
user_stats['risk_score'] = (
    (user_stats['attack_count']  / user_stats['attack_count'].max())  * 35 +
    (user_stats['unique_ips']    / user_stats['unique_ips'].max())    * 20 +
    (user_stats['bytes_total']   / user_stats['bytes_total'].max())   * 25 +
    user_stats['vpn_ratio']                                            * 10 +
    (user_stats['avg_attempts']  / user_stats['avg_attempts'].max())  * 10
).clip(0, 100)
user_stats['risk'] = user_stats['risk_score'].apply(risk)

# Platform nodes
plat_stats = atk.groupby('platform').agg(
    attack_count   = ('label',        'count'),
    unique_ips     = ('source_ip',    'nunique'),
    unique_users   = ('user',         'nunique'),
    bytes_total    = ('bytes_sent',   'sum'),
).reset_index()

# Edges
ip_user_edges  = atk.groupby(['source_ip','user']).agg(
    weight=('label','count'), bytes=('bytes_sent','sum'),
    atk_type=('attack_type', lambda x: x.value_counts().index[0])
).reset_index()
user_plat_edges = atk.groupby(['user','platform']).agg(
    weight=('label','count'), bytes=('bytes_sent','sum'),
    atk_type=('attack_type', lambda x: x.value_counts().index[0])
).reset_index()

# Top nodes for main graph
TOP_IP   = 12
TOP_USER = 18
top_ips   = ip_stats.nlargest(TOP_IP,   'threat_score')['source_ip'].tolist()
top_users = user_stats.nlargest(TOP_USER,'risk_score')['user'].tolist()

# Subgraph edges
sub_ip_user  = ip_user_edges[
    ip_user_edges['source_ip'].isin(top_ips) &
    ip_user_edges['user'].isin(top_users)]
sub_user_plat = user_plat_edges[user_plat_edges['user'].isin(top_users)]

# Edge width normalisation
def norm(s, lo=0.6, hi=5.0):
    mn, mx = s.min(), s.max()
    return lo if mn==mx else lo + (s-mn)/(mx-mn)*(hi-lo)

sub_ip_user  = sub_ip_user.copy();  sub_ip_user['lw']  = norm(sub_ip_user['weight'])
sub_user_plat= sub_user_plat.copy();sub_user_plat['lw'] = norm(sub_user_plat['weight'])

print(f"  Nodes  : {len(top_ips)} IPs  |  {len(top_users)} Users  |  2 Platforms")
print(f"  Edges  : {len(sub_ip_user)} IP→User  |  {len(sub_user_plat)} User→Platform")

# ── Layout ─────────────────────────────────────────────────────────────────────
# IPs: left column, staggered in two sub-columns
ip_pos = {}
half = TOP_IP // 2
for i, ip in enumerate(top_ips):
    col = i % 2
    row = i // 2
    ip_pos[ip] = (0.02 + col*0.11, 0.93 - row*(0.82/(half-1)) if half>1 else 0.5)

# Users: centre column, evenly spaced
user_pos = {}
for i, u in enumerate(top_users):
    user_pos[u] = (0.52, 0.95 - i*(0.90/(TOP_USER-1)))

# Platforms: right column
plat_pos = {'AWS': (0.93, 0.68), 'Azure': (0.93, 0.32)}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — 3 rows
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(26, 34), facecolor=BG)
fig.suptitle(
    '🕸   NETWORK GRAPH ANALYSIS  —  Attack Topology\n'
    'IP Attackers  ↔  Compromised / Targeted Users  ↔  Cloud Platforms',
    fontsize=16, fontweight='bold', color=TEXT, y=0.988, linespacing=2.0
)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       height_ratios=[2.8, 1.0, 1.0],
                       hspace=0.40, wspace=0.38,
                       left=0.04, right=0.97, top=0.965, bottom=0.02)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A  — Main Network Graph (full width, tall)
# ══════════════════════════════════════════════════════════════════════════════
ax_net = fig.add_subplot(gs[0, :])
ax_net.set_facecolor('#0A0E14')
for sp in ax_net.spines.values(): sp.set_edgecolor('#30363D'); sp.set_linewidth(1.5)
ax_net.set_xlim(-0.03, 1.06)
ax_net.set_ylim(-0.04, 1.06)
ax_net.axis('off')
ax_net.set_title('Full Attack Network  —  Top 12 IPs  →  18 Users  →  2 Platforms',
                 color=TEXT, fontsize=13, fontweight='bold', pad=12)

# --- Draw column header zones ---
for label, xc, yc, col in [
    ('SOURCE IPs\n(Attacker Nodes)', 0.065, 1.03, C_RED),
    ('TARGETED / COMPROMISED USERS', 0.52, 1.03, C_YEL),
    ('CLOUD PLATFORMS', 0.93, 1.03, C_GRN),
]:
    ax_net.text(xc, yc, label, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=col,
                transform=ax_net.transData)

# --- Edges: IP → User ---
for _, row in sub_ip_user.iterrows():
    if row['source_ip'] not in ip_pos or row['user'] not in user_pos: continue
    x0, y0 = ip_pos[row['source_ip']]
    x1, y1 = user_pos[row['user']]
    ec = ATK_PAL.get(row['atk_type'], C_BLUE)
    # Bezier-like curve via midpoint offset
    xm = (x0+x1)/2 + np.random.uniform(-0.03, 0.03)
    ym = (y0+y1)/2 + np.random.uniform(-0.02, 0.02)
    ax_net.annotate('', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle='->', color=ec,
            lw=float(row['lw'])*0.7,
            alpha=0.35,
            connectionstyle=f'arc3,rad={np.random.uniform(-0.15,0.15):.2f}'
        ))

# --- Edges: User → Platform ---
for _, row in sub_user_plat.iterrows():
    if row['user'] not in user_pos or row['platform'] not in plat_pos: continue
    x0, y0 = user_pos[row['user']]
    x1, y1 = plat_pos[row['platform']]
    ec = ATK_PAL.get(row['atk_type'], C_PUR)
    ax_net.annotate('', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle='->', color=ec,
            lw=float(row['lw'])*0.65,
            alpha=0.30,
            connectionstyle=f'arc3,rad={np.random.uniform(-0.2,0.2):.2f}'
        ))

# --- IP nodes ---
for ip, (x, y) in ip_pos.items():
    row = ip_stats[ip_stats['source_ip']==ip]
    if row.empty: continue
    ts   = row['threat_score'].values[0]
    risk_lv = row['risk'].values[0]
    rc   = RISK_PAL[risk_lv]
    sz   = 160 + ts * 3.5
    # Glow ring
    ax_net.scatter(x, y, s=sz*2.2, c=rc, alpha=0.10, zorder=4)
    ax_net.scatter(x, y, s=sz,     c=rc, alpha=0.90, zorder=5,
                   edgecolors='white', linewidths=0.6)
    # IP label
    ax_net.text(x, y+0.045, ip, ha='center', va='bottom',
                fontsize=6.2, fontfamily='monospace',
                color=rc, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='#0A0E14')])
    # Threat score badge
    ax_net.text(x, y-0.04, f'⚡{ts:.0f}', ha='center', va='top',
                fontsize=6.0, color='white', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='#0A0E14')])

# --- User nodes ---
for u, (x, y) in user_pos.items():
    row = user_stats[user_stats['user']==u]
    if row.empty: continue
    rs  = row['risk_score'].values[0]
    rlv = row['risk'].values[0]
    rc  = RISK_PAL[rlv]
    dom = row['dominant_attack'].values[0]
    # diamond shape via rotated square
    sz = 90 + rs * 2.5
    ax_net.scatter(x, y, s=sz*1.8, c=rc, alpha=0.12, zorder=4, marker='D')
    ax_net.scatter(x, y, s=sz,     c=rc, alpha=0.88, zorder=5,
                   marker='D', edgecolors='white', linewidths=0.5)
    ax_net.text(x+0.018, y, u, ha='left', va='center',
                fontsize=6.0, fontfamily='monospace', color=rc,
                path_effects=[pe.withStroke(linewidth=1.8, foreground='#0A0E14')])
    ax_net.text(x-0.018, y, f'{rs:.0f}', ha='right', va='center',
                fontsize=5.5, color='white', fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='#0A0E14')])

# --- Platform nodes ---
for plat, (x, y) in plat_pos.items():
    row = plat_stats[plat_stats['platform']==plat]
    cnt = row['attack_count'].values[0] if not row.empty else 0
    box = mpatches.FancyBboxPatch((x-0.055, y-0.055), 0.11, 0.11,
        boxstyle='round,pad=0.012',
        facecolor='#1F3A2A', edgecolor=C_GRN, linewidth=2.5, zorder=6, alpha=0.95)
    ax_net.add_patch(box)
    # Glow
    glow = mpatches.FancyBboxPatch((x-0.065, y-0.065), 0.13, 0.13,
        boxstyle='round,pad=0.015',
        facecolor='none', edgecolor=C_GRN, linewidth=5, zorder=5, alpha=0.15)
    ax_net.add_patch(glow)
    ax_net.text(x, y+0.008, plat, ha='center', va='center',
                fontsize=13, fontweight='bold', color=TEXT, zorder=7)
    ax_net.text(x, y-0.028, f'{cnt} attacks', ha='center', va='center',
                fontsize=7.5, color=C_GRN, zorder=7)

# --- Legend boxes ---
# Attack type legend
lx, ly = 0.01, 0.22
ax_net.add_patch(mpatches.FancyBboxPatch((lx-0.005, ly-0.005),
    0.20, len(ATK_PAL)*0.042+0.02,
    boxstyle='round,pad=0.008', facecolor='#0D1117',
    edgecolor=BORDER, linewidth=1, zorder=8, alpha=0.85))
ax_net.text(lx+0.09, ly+len(ATK_PAL)*0.042+0.008, 'ATTACK TYPE',
            ha='center', fontsize=7.5, fontweight='bold', color=MUTED, zorder=9)
for i, (atype, c) in enumerate(ATK_PAL.items()):
    yy = ly + i*0.042
    ax_net.scatter(lx+0.012, yy+0.012, s=60, c=c, zorder=9)
    ax_net.text(lx+0.028, yy+0.012, atype.replace('_',' '),
                va='center', fontsize=6.5, color=c, fontweight='bold', zorder=9)

# Risk legend
rx, ry = 0.01, 0.55
ax_net.add_patch(mpatches.FancyBboxPatch((rx-0.005, ry-0.005),
    0.18, len(RISK_PAL)*0.042+0.02,
    boxstyle='round,pad=0.008', facecolor='#0D1117',
    edgecolor=BORDER, linewidth=1, zorder=8, alpha=0.85))
ax_net.text(rx+0.085, ry+len(RISK_PAL)*0.042+0.008, 'RISK LEVEL',
            ha='center', fontsize=7.5, fontweight='bold', color=MUTED, zorder=9)
for i, (lv, c) in enumerate(RISK_PAL.items()):
    yy = ry + i*0.042
    ax_net.scatter(rx+0.012, yy+0.012, s=60, c=c, marker='o', zorder=9)
    ax_net.text(rx+0.028, yy+0.012, lv, va='center',
                fontsize=7, color=c, fontweight='bold', zorder=9)

# Node shape legend
sx, sy = 0.01, 0.79
ax_net.add_patch(mpatches.FancyBboxPatch((sx-0.005, sy-0.005),
    0.18, 0.14,
    boxstyle='round,pad=0.008', facecolor='#0D1117',
    edgecolor=BORDER, linewidth=1, zorder=8, alpha=0.85))
ax_net.text(sx+0.085, sy+0.128, 'NODE TYPE',
            ha='center', fontsize=7.5, fontweight='bold', color=MUTED, zorder=9)
ax_net.scatter(sx+0.018, sy+0.095, s=80, c=C_RED, marker='o', zorder=9)
ax_net.text(sx+0.035, sy+0.095, '● IP Attacker', va='center', fontsize=7, color=C_RED, zorder=9)
ax_net.scatter(sx+0.018, sy+0.055, s=80, c=C_YEL, marker='D', zorder=9)
ax_net.text(sx+0.035, sy+0.055, '◆ User Node',   va='center', fontsize=7, color=C_YEL, zorder=9)
ax_net.add_patch(mpatches.FancyBboxPatch((sx+0.006, sy+0.012), 0.025, 0.025,
    boxstyle='round,pad=0.003', facecolor=C_GRN, zorder=9, alpha=0.8))
ax_net.text(sx+0.035, sy+0.024, '■ Platform',    va='center', fontsize=7, color=C_GRN, zorder=9)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B  — IP Threat Score Ranking
# ══════════════════════════════════════════════════════════════════════════════
def mk_panel(ax, title):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(1)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    return ax

ax_b = mk_panel(fig.add_subplot(gs[1, 0]), '⚡  IP Threat Score Ranking')
top12 = ip_stats.nlargest(12, 'threat_score').sort_values('threat_score')
bar_c = [RISK_PAL[r] for r in top12['risk']]
bars = ax_b.barh(range(len(top12)), top12['threat_score'],
                 color=bar_c, height=0.65, edgecolor=BG)
ax_b.set_yticks(range(len(top12)))
ax_b.set_yticklabels([ip[-11:] for ip in top12['source_ip']],
                     fontsize=7.5, fontfamily='monospace', color=TEXT)
ax_b.set_xlabel('Threat Score', color=MUTED)
ax_b.set_xlim(0, 115)
for bar, (_, row) in zip(bars, top12.iterrows()):
    ax_b.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
              f"{row['threat_score']:.1f}  [{row['risk']}]",
              va='center', fontsize=7, fontweight='bold',
              color=RISK_PAL[row['risk']])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C  — User Risk Score Ranking
# ══════════════════════════════════════════════════════════════════════════════
ax_c = mk_panel(fig.add_subplot(gs[1, 1]), '👤  User Risk Score Ranking')
top18u = user_stats.nlargest(18, 'risk_score').sort_values('risk_score')
bar_c2 = [RISK_PAL[r] for r in top18u['risk']]
bars2 = ax_c.barh(range(len(top18u)), top18u['risk_score'],
                  color=bar_c2, height=0.65, edgecolor=BG)
ax_c.set_yticks(range(len(top18u)))
ax_c.set_yticklabels(top18u['user'].values,
                     fontsize=7.5, fontfamily='monospace', color=TEXT)
ax_c.set_xlabel('Risk Score', color=MUTED)
ax_c.set_xlim(0, 115)
for bar, (_, row) in zip(bars2, top18u.iterrows()):
    ax_c.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
              f"{row['risk_score']:.1f}",
              va='center', fontsize=7.5, fontweight='bold',
              color=RISK_PAL[row['risk']])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D  — Attack Type per IP heatmap
# ══════════════════════════════════════════════════════════════════════════════
ax_d = mk_panel(fig.add_subplot(gs[1, 2]), '🔥  Attack Type × Top IPs Heatmap')
ax_d.grid(False)
heat = atk[atk['source_ip'].isin(top_ips)].pivot_table(
    index='source_ip', columns='attack_type', values='label',
    aggfunc='count', fill_value=0
).astype(int)
heat.index = [ip[-12:] for ip in heat.index]
cmap_h = LinearSegmentedColormap.from_list('heat',
    ['#0D1117', '#1F2D1A', '#2EA043', '#FFA500', '#FF4444'])
import seaborn as sns
sns.heatmap(heat, ax=ax_d, cmap=cmap_h, linewidths=0.5,
            linecolor='#0D1117', annot=True, fmt='d',
            annot_kws={'size':7.5, 'color':TEXT, 'weight':'bold'},
            cbar_kws={'shrink':0.7})
ax_d.tick_params(colors=MUTED, labelsize=7)
ax_d.set_xlabel('Attack Type', color=MUTED, fontsize=8)
ax_d.set_ylabel('Source IP',   color=MUTED, fontsize=8)
ax_d.set_xticklabels([t.get_text().replace('_','\n') for t in ax_d.get_xticklabels()],
                     fontsize=6.5)
ax_d.collections[0].colorbar.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(ax_d.collections[0].colorbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=7)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL E  — Platform attack breakdown stacked bar
# ══════════════════════════════════════════════════════════════════════════════
ax_e = mk_panel(fig.add_subplot(gs[2, 0]), '☁️  Platform  ×  Attack Type Breakdown')
pt = atk.groupby(['platform','attack_type']).size().unstack(fill_value=0)
pt_norm = pt.div(pt.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(pt_norm))
atk_types = pt_norm.columns.tolist()
for at in atk_types:
    vals = pt_norm[at].values
    bars_e = ax_e.bar(pt_norm.index, vals, bottom=bottom,
                      label=at.replace('_',' '), color=ATK_PAL.get(at, C_BLUE),
                      edgecolor=BG, linewidth=0.8)
    # labels inside bar
    for rect, v, b in zip(bars_e, vals, bottom):
        if v > 4:
            ax_e.text(rect.get_x()+rect.get_width()/2, b+v/2,
                      f'{v:.0f}%', ha='center', va='center',
                      fontsize=7, fontweight='bold', color=BG)
    bottom += vals
ax_e.set_ylabel('Share (%)', color=MUTED)
ax_e.set_ylim(0, 115)
ax_e.tick_params(colors=MUTED)
ax_e.legend(loc='upper right', fontsize=6.5, ncol=1)
# raw counts
for i, (plat, row) in enumerate(pt.iterrows()):
    ax_e.text(i, 102, f'n={row.sum()}', ha='center',
              fontsize=8, fontweight='bold', color=TEXT)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL F  — IP × User edge weight matrix (top 8 × 10)
# ══════════════════════════════════════════════════════════════════════════════
ax_f = mk_panel(fig.add_subplot(gs[2, 1]), '🔗  IP → User Edge Weight Matrix')
ax_f.grid(False)
top8_ip   = ip_stats.nlargest(8,  'threat_score')['source_ip'].tolist()
top10_usr = user_stats.nlargest(10,'risk_score')['user'].tolist()
matrix = ip_user_edges[
    ip_user_edges['source_ip'].isin(top8_ip) &
    ip_user_edges['user'].isin(top10_usr)
].pivot_table(index='source_ip', columns='user', values='weight', fill_value=0).astype(int)
matrix.index   = [ip[-12:] for ip in matrix.index]
matrix.columns = [u for u in matrix.columns]
cmap_m = LinearSegmentedColormap.from_list('edge',
    ['#161B22','#1F3A5F','#58A6FF','#E3B341','#FF4444'])
sns.heatmap(matrix, ax=ax_f, cmap=cmap_m, linewidths=0.6,
            linecolor='#0D1117', annot=True, fmt='d',
            annot_kws={'size':8, 'color':TEXT, 'weight':'bold'},
            cbar_kws={'shrink':0.7})
ax_f.tick_params(colors=MUTED, labelsize=7)
ax_f.set_xlabel('User', color=MUTED, fontsize=8)
ax_f.set_ylabel('Source IP', color=MUTED, fontsize=8)
plt.setp(ax_f.get_xticklabels(), rotation=35, ha='right', fontsize=6.5)
ax_f.collections[0].colorbar.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(ax_f.collections[0].colorbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=7)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL G  — Top Attack Chains (IP → User → Platform)
# ══════════════════════════════════════════════════════════════════════════════
ax_g = mk_panel(fig.add_subplot(gs[2, 2]), '⛓️  Top 10 Full Attack Chains')
ax_g.axis('off')
ax_g.grid(False)

# Build chains: ip → user → platform (top by combined weight)
chains = atk.merge(
    ip_user_edges[['source_ip','user','weight']].rename(columns={'weight':'w1'}),
    on=['source_ip','user'], how='left'
).groupby(['source_ip','user','platform']).agg(
    total=('label','count'),
    bytes=('bytes_sent','sum'),
    atk_type=('attack_type', lambda x: x.value_counts().index[0])
).reset_index().nlargest(10,'total')

# Table-style render
headers = ['Rank','Source IP','User','Platform','Hits','Type']
col_x   = [0.00, 0.07, 0.37, 0.58, 0.72, 0.80]
col_w   = TEXT

# Header row
for hdr, cx in zip(headers, col_x):
    ax_g.text(cx, 0.96, hdr, fontsize=7.5, fontweight='bold',
              color=MUTED, va='top', transform=ax_g.transAxes)
ax_g.axhline(y=0.935, xmin=0.0, xmax=1.0, color=BORDER,
             linewidth=1)

for i, (_, row) in enumerate(chains.iterrows()):
    y = 0.89 - i*0.084
    bg_c = '#1A1D27' if i%2==0 else '#161B22'
    ax_g.add_patch(mpatches.FancyBboxPatch(
        (0.0, y-0.032), 1.0, 0.068,
        boxstyle='round,pad=0.005',
        facecolor=bg_c, edgecolor='none',
        transform=ax_g.transAxes, zorder=1))
    atk_c = ATK_PAL.get(row['atk_type'], C_BLUE)
    vals = [
        f"#{i+1}",
        row['source_ip'][-13:],
        row['user'],
        row['platform'],
        str(int(row['total'])),
        row['atk_type'].replace('_','\n'),
    ]
    for val, cx in zip(vals, col_x):
        fc = (C_RED if cx==col_x[1] else
              C_YEL if cx==col_x[2] else
              C_GRN if cx==col_x[3] else
              atk_c if cx==col_x[5] else TEXT)
        ax_g.text(cx, y+0.005, val,
                  fontsize=6.2, color=fc, va='center',
                  fontfamily=('monospace' if cx in col_x[:3] else 'sans-serif'),
                  fontweight='bold' if cx in [col_x[0], col_x[4]] else 'normal',
                  transform=ax_g.transAxes, zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
plt.savefig('../output/network_graph.png', dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PRINT FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  NETWORK GRAPH FINDINGS")
print("="*65)

print(f"\n  TOP 5 THREAT IPs:")
for _, r in ip_stats.nlargest(5,'threat_score').iterrows():
    print(f"  {RISK_PAL[r['risk']].replace('#','')} "
          f"{r['source_ip']:<22}  score={r['threat_score']:.1f}"
          f"  users={r['unique_users']}  [{r['risk']}]  → {r['dominant_attack']}")

print(f"\n  TOP 5 COMPROMISED USERS:")
for _, r in user_stats.nlargest(5,'risk_score').iterrows():
    print(f"  {r['user']:<15}  score={r['risk_score']:.1f}"
          f"  ips={r['unique_ips']}  [{r['risk']}]  → {r['dominant_attack']}")

print(f"\n  PLATFORM EXPOSURE:")
for _, r in plat_stats.iterrows():
    print(f"  {r['platform']:<8}  attacks={r['attack_count']}  "
          f"unique_ips={r['unique_ips']}  unique_users={r['unique_users']}")

print(f"\n  TOP 3 ATTACK CHAINS (IP → User → Platform):")
for i, (_, r) in enumerate(chains.head(3).iterrows()):
    print(f"  {i+1}. {r['source_ip']} → {r['user']} → {r['platform']}"
          f"  ({int(r['total'])} hits, {r['atk_type']})")

print(f"\n  KEY INSIGHTS:")
# Multi-user IP
multi_user_ips = ip_stats[ip_stats['unique_users']>2]
print(f"  • {len(multi_user_ips)} IPs targeted >2 users (coordinated campaign indicator)")
# Multi-platform users
multi_plat_u = user_stats[user_stats['unique_platforms']>1]
print(f"  • {len(multi_plat_u)} users attacked across both AWS & Azure")
# VPN-heavy
vpn_heavy = ip_stats[ip_stats['vpn_ratio']>0.6]
print(f"  • {len(vpn_heavy)} IPs used VPN >60% of the time (evasion behavior)")
# Data exfil IPs
exfil = ip_stats[ip_stats['dominant_attack']=='DATA_EXFILTRATION']
print(f"  • {len(exfil)} IPs primarily conducting DATA EXFILTRATION")

print("\n" + "="*65)
print("  ✅  Network Graph Analysis Complete!")
print("="*65)
