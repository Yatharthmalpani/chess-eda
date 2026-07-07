# ============================================================
#   CHESS GAMES - EXPLORATORY DATA ANALYSIS 
#   Dataset : games.csv (~20,000 Lichess games)
#   Tools   : numpy, pandas, matplotlib, seaborn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ============================================================
# STEP 1 — LOAD & CLEAN DATA
# ============================================================

df = pd.read_csv("games.csv")

print("Shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nWinner counts:\n",        df["winner"].value_counts())
print("\nVictory status counts:\n", df["victory_status"].value_counts())

# Create useful columns
df["avg_rating"]    = (df["white_rating"] + df["black_rating"]) / 2
df["rating_diff"]   = df["white_rating"] - df["black_rating"]
df["abs_diff"]      = df["rating_diff"].abs()

BINS   = [0, 1000, 1200, 1400, 1600, 1800, 2000, 3000]
LABELS = ["<1000","1000-1200","1200-1400","1400-1600","1600-1800","1800-2000","2000+"]
df["rating_group"] = pd.cut(df["avg_rating"], bins=BINS, labels=LABELS)

total = len(df)
print(f"\nTotal games : {total:,}")
print(f"White wins  : {(df['winner']=='white').sum():,}  ({(df['winner']=='white').mean()*100:.1f}%)")
print(f"Black wins  : {(df['winner']=='black').sum():,}  ({(df['winner']=='black').mean()*100:.1f}%)")
print(f"Draws       : {(df['winner']=='draw').sum():,}  ({(df['winner']=='draw').mean()*100:.1f}%)")


# ============================================================
# FIGURE 1 — Who wins? Overall outcome breakdown  (1×3)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("WHO WINS? — OVERALL OUTCOME BREAKDOWN", fontsize=12)
plt.subplots_adjust(wspace=0.38)

# 1-A  Pie chart — simple outcome split
ax = axes[0]
sizes  = [
    (df["winner"] == "white").sum(),
    (df["winner"] == "black").sum(),
    (df["winner"] == "draw" ).sum(),
]
labels = ["White Wins", "Black Wins", "Draw"]
colors = ['blue', 'orange', 'gray']
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1),
)
ax.set_title("Outcome Split (Pie)")

# 1-B  Bar chart — same data as bars
ax = axes[1]
bars = ax.bar(labels, [s / total * 100 for s in sizes],
              color=colors, edgecolor='black', linewidth=1, width=0.5)
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
            f"{height:.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 65)
ax.set_title("Outcome Split (Bar)")

# 1-C  Victory status — how games actually end
ax = axes[2]
vs_counts = df["victory_status"].value_counts()
pal = ['red', 'blue', 'orange', 'gray']
bars2 = ax.bar(vs_counts.index, vs_counts.values,
               color=pal[:len(vs_counts)], edgecolor='black', linewidth=1)
# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 50,
            f"{int(height)}", ha="center", va="bottom", fontsize=8)
ax.set_title("How Games End (Victory Status)")
ax.set_ylabel("Number of Games")

plt.tight_layout()
plt.savefig("fig1_outcome_overview.png")
plt.close()
print("Saved fig1_outcome_overview.png")


# ============================================================
# FIGURE 2 — Does White have an unfair advantage?  (2×2)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DOES WHITE HAVE AN ADVANTAGE?", fontsize=12)
plt.subplots_adjust(hspace=0.4, wspace=0.35)

# 2-A  White win rate per rating group (line chart)
ax = axes[0, 0]
white_rates = []
for grp in LABELS:
    sub = df[df["rating_group"] == grp]
    n   = len(sub[sub["winner"].isin(["white","black","draw"])])
    rate = (sub["winner"] == "white").sum() / n * 100 if n > 0 else np.nan
    white_rates.append(rate)

ax.plot(LABELS, white_rates, color='blue', lw=2, marker="o", markersize=6)
ax.axhline(50, color='gray', lw=1, ls="--", alpha=0.6, label="50% line")
ax.fill_between(LABELS, white_rates, 50,
                where=[w > 50 for w in white_rates],
                alpha=0.15, color='blue')
ax.set_xticks(range(len(LABELS)))
ax.set_xticklabels(LABELS, rotation=30, ha="right")
ax.set_ylabel("White Win Rate (%)")
ax.set_ylim(40, 65)
ax.legend(fontsize=8)
ax.set_title("White Win Rate by Rating Group")

# 2-B  Stacked bar — White / Black / Draw % per rating group
ax = axes[0, 1]
w_pcts, b_pcts, d_pcts = [], [], []
for grp in LABELS:
    sub = df[df["rating_group"] == grp]
    n   = max(len(sub), 1)
    w_pcts.append((sub["winner"] == "white").sum() / n * 100)
    b_pcts.append((sub["winner"] == "black").sum() / n * 100)
    d_pcts.append((sub["winner"] == "draw" ).sum() / n * 100)

x = np.arange(len(LABELS))
ax.bar(x, w_pcts, label="White",  color='blue', edgecolor='black', linewidth=0.5)
ax.bar(x, b_pcts, label="Black",  color='orange', edgecolor='black', linewidth=0.5, bottom=w_pcts)
bottom2 = [a + b for a, b in zip(w_pcts, b_pcts)]
ax.bar(x, d_pcts, label="Draw",   color='gray',  edgecolor='black', linewidth=0.5, bottom=bottom2)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=30, ha="right")
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 105)
ax.legend(fontsize=8)
ax.set_title("Outcome % per Rating Group (Stacked)")

# 2-C  Draw rate by rating group
ax = axes[1, 0]
draw_rates = [d for d in d_pcts]
bars3 = ax.bar(LABELS, draw_rates, color='gray', edgecolor='black', linewidth=1)
for bar in bars3:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
            f"{height:.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(LABELS)))
ax.set_xticklabels(LABELS, rotation=30, ha="right")
ax.set_ylabel("Draw Rate (%)")
ax.set_title("Draw Rate by Rating Group")

# 2-D  Rated vs Unrated — does it affect who wins?
ax = axes[1, 1]
rated_df   = df[df["rated"] == True]
unrated_df = df[df["rated"] == False]
cat_labels  = ["White Wins", "Black Wins", "Draw"]
rated_vals   = [(rated_df["winner"] == w).mean() * 100   for w in ["white","black","draw"]]
unrated_vals = [(unrated_df["winner"] == w).mean() * 100 for w in ["white","black","draw"]]
x = np.arange(3)
w = 0.35
ax.bar(x - w/2, rated_vals,   w, label="Rated",   color='red',  edgecolor='black')
ax.bar(x + w/2, unrated_vals, w, label="Unrated", color='purple',  edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(cat_labels)
ax.set_ylabel("Percentage (%)")
ax.legend(fontsize=8)
ax.set_title("Rated vs Unrated Games")

plt.tight_layout()
plt.savefig("fig2_white_advantage.png")
plt.close()
print("Saved fig2_white_advantage.png")


# ============================================================
# FIGURE 3 — Rating & Winning  (2×2)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DOES RATING ACTUALLY MATTER?", fontsize=12)
plt.subplots_adjust(hspace=0.4, wspace=0.35)

games = df[df["winner"].isin(["white","black"])].copy()
games["higher_rated_wins"] = (
    ((games["white_rating"] > games["black_rating"]) & (games["winner"] == "white")) |
    ((games["black_rating"] > games["white_rating"]) & (games["winner"] == "black"))
).astype(int)

# 3-A  Win probability by rating gap bucket
ax = axes[0, 0]
gap_bins   = [0, 50, 100, 200, 300, 500, 1500]
gap_labels = ["0–50","50–100","100–200","200–300","300–500","500+"]
games["gap_bin"] = pd.cut(games["abs_diff"], bins=gap_bins, labels=gap_labels)

pred  = games.groupby("gap_bin", observed=True)["higher_rated_wins"].mean() * 100
count = games.groupby("gap_bin", observed=True)["higher_rated_wins"].count()

ax.plot(gap_labels, pred.values, color='green', lw=2, marker="D", markersize=7)
ax.fill_between(gap_labels, pred.values, 50, alpha=0.2, color='green')
ax.axhline(50, color='gray', lw=1, ls="--", alpha=0.6, label="50% (random)")
for i, (val, cnt) in enumerate(zip(pred.values, count.values)):
    ax.text(i, val + 1.5, f"n={cnt:,}", ha="center", fontsize=7, color='gray')
ax.set_ylabel("Higher-Rated Player Win Rate (%)")
ax.set_ylim(40, 100)
ax.legend(fontsize=8)
ax.set_title("How Much Rating Gap Predicts the Winner")

# 3-B  Histogram — rating difference distribution
ax = axes[0, 1]
ax.hist(games["rating_diff"], bins=60, color='purple', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.axvline(0, color='red', lw=2, ls="--", label="Zero difference")
ax.axvline(games["rating_diff"].mean(), color='blue', lw=1.5, ls=":",
           label=f"Mean = {games['rating_diff'].mean():.0f}")
ax.set_xlabel("Rating Difference (White − Black)")
ax.set_ylabel("Number of Games")
ax.legend(fontsize=8)
ax.set_title("Rating Difference Distribution")

# 3-C  Boxplot — ratings of winners vs losers
ax = axes[1, 0]
winner_ratings  = df[df["winner"] == "white"]["white_rating"].values
loser_ratings   = df[df["winner"] == "black"]["white_rating"].values   # white lost
winner_b_rating = df[df["winner"] == "black"]["black_rating"].values
loser_b_rating  = df[df["winner"] == "white"]["black_rating"].values   # black lost

all_winner = np.concatenate([winner_ratings, winner_b_rating])
all_loser  = np.concatenate([loser_ratings,  loser_b_rating])

bp = ax.boxplot([all_winner, all_loser], patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor('green')
bp["boxes"][0].set_alpha(0.4)
bp["boxes"][1].set_facecolor('red')
bp["boxes"][1].set_alpha(0.4)

ax.set_xticklabels(["Winners","Losers"], fontsize=9)
ax.set_ylabel("Player Rating")
ax.set_title("Rating of Winners vs Losers")

# 3-D  White vs Black rating distribution (violin)
ax = axes[1, 1]
data_violin = [df["white_rating"].values, df["black_rating"].values]
vp = ax.violinplot(data_violin, positions=[1, 2], showmedians=True, showextrema=False)
colors_violin = ['blue', 'orange']
for body, col in zip(vp["bodies"], colors_violin):
    body.set_facecolor(col)
    body.set_alpha(0.4)
vp["cmedians"].set_color('black')
vp["cmedians"].set_linewidth(1.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(["White Rating","Black Rating"], fontsize=9)
ax.set_ylabel("Rating")
ax.set_title("Rating Distribution: White vs Black")

plt.tight_layout()
plt.savefig("fig3_rating_matters.png")
plt.close()
print("Saved fig3_rating_matters.png")


# ============================================================
# FIGURE 4 — Openings Analysis  (2×2)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle("CHESS OPENINGS — POPULARITY & EFFECTIVENESS", fontsize=12)
plt.subplots_adjust(hspace=0.5, wspace=0.45, top=0.94)

# 4-A  Top 20 most played openings
ax = axes[0, 0]
top20 = df["opening_name"].value_counts().head(20)
ax.barh(range(20), top20.values[::-1], color='blue', edgecolor='black', linewidth=0.5, alpha=0.7)
ax.set_yticks(range(20))
ax.set_yticklabels([n[:36] for n in top20.index[::-1]], fontsize=7)
for i, val in enumerate(top20.values[::-1]):
    ax.text(val + 2, i, str(val), va="center", color='black', fontsize=7)
ax.set_xlabel("Number of Games")
ax.set_title("Top 20 Most Popular Openings")

# 4-B  Top 15 openings by WHITE win rate  (min 100 games)
ax = axes[0, 1]
op = (df.groupby("opening_name")
        .agg(games=("winner","count"),
             white_wins=("winner", lambda x: (x=="white").sum()))
        .reset_index())
op = op[op["games"] >= 100].copy()
op["white_win_pct"] = op["white_wins"] / op["games"] * 100
top_wr = op.sort_values("white_win_pct", ascending=False).head(15)

ax.barh(range(15), top_wr["white_win_pct"].values[::-1], color='orange', edgecolor='black', linewidth=0.5, alpha=0.7)
ax.set_yticks(range(15))
ax.set_yticklabels([n[:36] for n in top_wr["opening_name"].values[::-1]], fontsize=7)
ax.axvline(50, color='gray', lw=1, ls="--", alpha=0.7, label="50%")
for i, val in enumerate(top_wr["white_win_pct"].values[::-1]):
    ax.text(val + 0.4, i, f"{val:.1f}%", va="center", color='black', fontsize=7)
ax.set_xlim(0, 82)
ax.set_xlabel("White Win Rate (%)")
ax.legend(fontsize=8)
ax.set_title("Best Openings for White (min 100 games)")

# 4-C  Scatter — Popularity vs Win Rate
ax = axes[1, 0]
op50 = op[op["games"] >= 50].copy()
scatter = ax.scatter(op50["games"], op50["white_win_pct"],
                     c=op50["white_win_pct"], cmap="coolwarm",
                     s=op50["games"] / 5, alpha=0.7,
                     edgecolors='black', linewidths=0.5)
ax.axhline(50, color='gray', lw=1, ls="--", alpha=0.6)
# label the most popular ones
for _, row in op50.nlargest(6, "games").iterrows():
    ax.annotate(row["opening_name"][:22],
                (row["games"], row["white_win_pct"]),
                fontsize=6, color='black', alpha=0.8,
                xytext=(5, 3), textcoords="offset points")
plt.colorbar(scatter, ax=ax, label="White Win %")
ax.set_xlabel("Number of Games (Popularity)")
ax.set_ylabel("White Win Rate (%)")
ax.set_title("Are Popular Openings Actually Effective?")

# 4-D  Opening diversity — cumulative coverage
ax = axes[1, 1]
counts = df["opening_name"].value_counts()
cumsum = counts.cumsum() / counts.sum() * 100
x_vals = range(1, min(101, len(cumsum)+1))
ax.plot(x_vals, cumsum.values[:100], color='red', lw=2)
ax.axhline(cumsum.iloc[9],  color='blue', lw=1, ls="--",
           label=f"Top 10 → {cumsum.iloc[9]:.1f}%")
ax.axhline(cumsum.iloc[19], color='purple',  lw=1, ls="--",
           label=f"Top 20 → {cumsum.iloc[19]:.1f}%")
ax.fill_between(x_vals, cumsum.values[:100], alpha=0.1, color='red')
ax.axvline(10, color='blue', lw=0.8, ls="--", alpha=0.6)
ax.axvline(20, color='purple',  lw=0.8, ls="--", alpha=0.6)
ax.set_xlabel("Top N Openings")
ax.set_ylabel("Cumulative % of All Games")
ax.legend(fontsize=8)
ax.set_title(f"Opening Diversity ({counts.shape[0]} unique openings)")

plt.tight_layout()
plt.savefig("fig4_openings.png")
plt.close()
print("Saved fig4_openings.png")


# ============================================================
# FIGURE 5 — Game Length  (2×2)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HOW LONG DO CHESS GAMES LAST?", fontsize=12)
plt.subplots_adjust(hspace=0.4, wspace=0.35)

# 5-A  Histogram of game length (turns)
ax = axes[0, 0]
ax.hist(df["turns"], bins=70, color='orange', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.axvline(df["turns"].median(), color='blue', lw=2, ls="--",
           label=f"Median = {df['turns'].median():.0f} turns")
ax.axvline(df["turns"].mean(), color='red', lw=2, ls=":",
           label=f"Mean = {df['turns'].mean():.0f} turns")
ax.set_xlabel("Number of Turns")
ax.set_ylabel("Number of Games")
ax.legend(fontsize=8)
ax.set_title("Distribution of Game Length")

# 5-B  Boxplot — turns by winner
ax = axes[0, 1]
data_turns = [
    df.loc[df["winner"] == "white", "turns"].values,
    df.loc[df["winner"] == "black", "turns"].values,
    df.loc[df["winner"] == "draw",  "turns"].values,
]
bp = ax.boxplot(data_turns, patch_artist=True, widths=0.45)
colors_turns = ['blue', 'orange', 'gray']
for box, col in zip(bp["boxes"], colors_turns):
    box.set_facecolor(col)
    box.set_alpha(0.4)
ax.set_xticklabels(["White Wins","Black Wins","Draw"], fontsize=8)
ax.set_ylabel("Number of Turns")
ax.set_title("Game Length by Winner")

# 5-C  White win rate by game-length bucket
ax = axes[1, 0]
turn_bins   = [0, 20, 40, 60, 80, 120, 400]
turn_labels = ["0–20","20–40","40–60","60–80","80–120","120+"]
df["turn_group"] = pd.cut(df["turns"], bins=turn_bins, labels=turn_labels)
wrate_turns = []
count_turns = []
for tg in turn_labels:
    sub = df[df["turn_group"] == tg]
    n   = max(len(sub[sub["winner"].isin(["white","black","draw"])]), 1)
    wrate_turns.append((sub["winner"] == "white").sum() / n * 100)
    count_turns.append(len(sub))

ax.plot(turn_labels, wrate_turns, color='blue', lw=2, marker="s", markersize=6)
ax.axhline(50, color='gray', lw=1, ls="--", alpha=0.6, label="50% line")
for i, (val, cnt) in enumerate(zip(wrate_turns, count_turns)):
    ax.text(i, val + 1.2, f"n={cnt:,}", ha="center", fontsize=7, color='gray')
ax.set_ylabel("White Win Rate (%)")
ax.set_ylim(35, 70)
ax.legend(fontsize=8)
ax.set_title("Do Short Games Favour White?")

# 5-D  Average game length by victory status
ax = axes[1, 1]
avg_turns = df.groupby("victory_status")["turns"].mean().sort_values(ascending=False)
pal2 = ['red', 'blue', 'orange', 'gray']
bars4 = ax.bar(avg_turns.index, avg_turns.values,
               color=pal2[:len(avg_turns)], edgecolor='black', linewidth=1)
for bar in bars4:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.4,
            f"{height:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Average Turns")
ax.set_title("Average Game Length by How It Ended")

plt.tight_layout()
plt.savefig("fig5_game_length.png")
plt.close()
print("Saved fig5_game_length.png")


# ============================================================
# FIGURE 6 — Beginner vs Expert Behaviour  (2×3)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("BEGINNER vs EXPERT — HOW DIFFERENTLY DO THEY PLAY?", fontsize=12)
plt.subplots_adjust(hspace=0.45, wspace=0.38, top=0.93)

buckets = ["<1200","1200–1600","1600–2000","2000+"]
b_bins  = [0, 1200, 1600, 2000, 3000]
df["skill_group"] = pd.cut(df["avg_rating"], bins=b_bins, labels=buckets)

# 6-A  Average turns by skill group
ax = axes[0, 0]
avg_t = df.groupby("skill_group", observed=True)["turns"].mean()
bars5 = ax.bar(buckets, avg_t.values,
               color=['purple', 'blue', 'green', 'orange'],
               edgecolor='black', linewidth=1)
for bar in bars5:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.4,
            f"{height:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Average Number of Turns")
ax.set_title("Do Experts Play Longer Games?")

# 6-B  Resign % vs Timeout % by skill group
ax = axes[0, 1]
resign_pct  = [
    (df[df["skill_group"] == g]["victory_status"] == "resign").mean() * 100
    for g in buckets
]
timeout_pct = [
    (df[df["skill_group"] == g]["victory_status"] == "outoftime").mean() * 100
    for g in buckets
]
x = np.arange(len(buckets))
w = 0.35
ax.bar(x - w/2, resign_pct,  w, label="Resign %",  color='red',  edgecolor='black')
ax.bar(x + w/2, timeout_pct, w, label="Timeout %", color='blue', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(buckets)
ax.set_ylabel("Percentage (%)")
ax.legend(fontsize=8)
ax.set_title("Resign vs Timeout by Skill Level")

# 6-C  Favorite opening per skill group
ax = axes[0, 2]
fav = {}
for grp in buckets:
    sub = df[df["skill_group"] == grp]
    fav[grp] = sub["opening_name"].value_counts().head(1).index[0]

fav_counts = []
for grp in buckets:
    sub = df[df["skill_group"] == grp]
    fav_counts.append(sub[sub["opening_name"] == fav[grp]].shape[0])

colors_fav = ['purple', 'blue', 'green', 'orange']
bars6 = ax.bar(buckets, fav_counts, color=colors_fav, edgecolor='black', linewidth=1)
ax.set_ylabel("Games Played with That Opening")
ax.set_title("Top Opening per Skill Group")
for bar, grp in zip(bars6, buckets):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            fav[grp][:20], ha="center", fontsize=6, color='black', rotation=8)

# 6-D  Rating distribution per skill group
ax = axes[1, 0]
colors_sk = ['purple', 'blue', 'green', 'orange']
for grp, col in zip(buckets, colors_sk):
    sub = df[df["skill_group"] == grp]["avg_rating"].dropna()
    ax.hist(sub, bins=30, color=col, alpha=0.5, label=grp, edgecolor="none")
ax.set_xlabel("Average Rating")
ax.set_ylabel("Number of Games")
ax.legend(fontsize=8)
ax.set_title("Rating Distribution by Skill Group")

# 6-E  Mate % vs Resign % vs Draw % — full breakdown per skill group
ax = axes[1, 1]
statuses = ["mate","resign","outoftime","draw"]
status_colors = ['orange', 'red', 'blue', 'gray']
bottom_vals = np.zeros(len(buckets))
for status, col in zip(statuses, status_colors):
    vals = [
        (df[df["skill_group"] == g]["victory_status"] == status).mean() * 100
        for g in buckets
    ]
    ax.bar(buckets, vals, label=status.capitalize(),
           color=col, edgecolor='black', linewidth=0.5, bottom=bottom_vals)
    bottom_vals += np.array(vals)
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 105)
ax.legend(fontsize=8)
ax.set_title("Victory Status Breakdown by Skill")

# 6-F  Number of games per skill group
ax = axes[1, 2]
grp_counts = df["skill_group"].value_counts().reindex(buckets)
bars7 = ax.bar(buckets, grp_counts.values,
               color=colors_sk, edgecolor='black', linewidth=1)
for bar in bars7:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 30,
            f"{int(height)}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Number of Games")
ax.set_title("How Many Games per Skill Group?")

plt.tight_layout()
plt.savefig("fig6_beginner_vs_expert.png")
plt.close()
print("Saved fig6_beginner_vs_expert.png")


# ============================================================
# FIGURE 7 — Fun & Interesting Questions  (2×3)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("INTERESTING QUESTIONS — PATTERNS & SURPRISES", fontsize=12)
plt.subplots_adjust(hspace=0.45, wspace=0.4, top=0.93)

# 7-A  Which opening has the most draws?
ax = axes[0, 0]
op_draw = (df.groupby("opening_name")
             .agg(games=("winner","count"),
                  draws=("winner", lambda x: (x=="draw").sum()))
             .reset_index())
op_draw = op_draw[op_draw["games"] >= 80].copy()
op_draw["draw_pct"] = op_draw["draws"] / op_draw["games"] * 100
top_draw = op_draw.sort_values("draw_pct", ascending=False).head(12)

ax.barh(range(12), top_draw["draw_pct"].values[::-1],
        color='gray', edgecolor='black', linewidth=0.5, alpha=0.7)
ax.set_yticks(range(12))
ax.set_yticklabels([n[:32] for n in top_draw["opening_name"].values[::-1]], fontsize=7)
ax.set_xlabel("Draw Rate (%)")
ax.set_title("Which Openings End in Draws Most? (min 80 games)")

# 7-B  Quickest checkmates — turn count distribution for "mate" wins
ax = axes[0, 1]
mates = df[df["victory_status"] == "mate"]["turns"]
ax.hist(mates, bins=50, color='red', edgecolor='black', linewidth=0.3, alpha=0.8)
ax.axvline(mates.median(), color='blue', lw=2, ls="--",
           label=f"Median = {mates.median():.0f} turns")
ax.axvline(20, color='green', lw=1.5, ls=":",
           label=f"<20 turns: {(mates<20).sum()} games")
ax.set_xlabel("Number of Turns")
ax.set_ylabel("Number of Games")
ax.legend(fontsize=8)
ax.set_title("How Many Turns Before Checkmate?")

# 7-C  Top openings preferred by winners
ax = axes[0, 2]
winners_df = df[df["winner"].isin(["white","black"])].copy()

top_winner_openings = winners_df["opening_name"].value_counts().head(15)
wr_for_top = []
for op_name in top_winner_openings.index:
    sub = df[df["opening_name"] == op_name]
    n   = max(len(sub[sub["winner"].isin(["white","black","draw"])]), 1)
    wr  = (sub["winner"] == "white").sum() / n * 100
    wr_for_top.append(wr)

colors_top = ['green' if w >= 50 else 'red' for w in wr_for_top]
ax.barh(range(15), wr_for_top[::-1],
        color=colors_top[::-1], edgecolor='black', linewidth=0.5, alpha=0.7)
ax.set_yticks(range(15))
ax.set_yticklabels([n[:32] for n in top_winner_openings.index[::-1]], fontsize=7)
ax.axvline(50, color='gray', lw=1, ls="--", alpha=0.7)
ax.set_xlabel("White Win Rate (%)")
ax.set_title("White Win Rate in Popular Openings")

# 7-D  Blunder games — very short games (≤15 turns) who wins?
ax = axes[1, 0]
short = df[df["turns"] <= 15]
if len(short) > 0:
    sc = short["winner"].value_counts()
    bars_short = ax.bar(sc.index, sc.values,
                        color=['blue' if v=="white" else 'orange' if v=="black" else 'gray' for v in sc.index],
                        edgecolor='black', linewidth=1)
    for bar in bars_short:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                str(int(height)), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Number of Games")
ax.set_title(f"Very Short Games (≤15 turns) — Who Wins?")

# 7-E  Correlation heatmap — numeric columns
ax = axes[1, 1]
num_cols_corr = ["white_rating","black_rating","turns","avg_rating","abs_diff"]
corr = df[num_cols_corr].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            linewidths=0.5, linecolor='white')
ax.set_title("Correlation Between Numeric Columns")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

# 7-F  Are longer openings (more opening moves) better for White?
ax = axes[1, 2]
df["opening_length"] = df["opening_ply"]
olen_bins   = [0, 2, 4, 6, 8, 10, 30]
olen_labels = ["1–2","3–4","5–6","7–8","9–10","11+"]
df["olen_group"] = pd.cut(df["opening_length"], bins=olen_bins, labels=olen_labels)

wr_olen = []
count_olen = []
for og in olen_labels:
    sub = df[df["olen_group"] == og]
    n   = max(len(sub[sub["winner"].isin(["white","black","draw"])]), 1)
    wr_olen.append((sub["winner"] == "white").sum() / n * 100)
    count_olen.append(len(sub))

bars_olen = ax.bar(olen_labels, wr_olen, color='orange', edgecolor='black', linewidth=1, alpha=0.8)
ax.axhline(50, color='gray', lw=1, ls="--", alpha=0.6)
for i, (val, cnt) in enumerate(zip(wr_olen, count_olen)):
    ax.text(i, val + 0.5, f"{val:.1f}%\nn={cnt:,}", ha="center", fontsize=7)
ax.set_xlabel("Opening Length (half-moves / ply)")
ax.set_ylabel("White Win Rate (%)")
ax.set_ylim(35, 65)
ax.set_title("Does a Longer Opening Help White?")

plt.tight_layout()
plt.savefig("fig7_interesting_questions.png")
plt.close()
print("Saved fig7_interesting_questions.png")


# ============================================================
# FIGURE 8 — Opening ECO Codes  (1×2)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("ECO CODE ANALYSIS — CHESS OPENING FAMILIES", fontsize=12, y=1.01)
plt.subplots_adjust(wspace=0.4)

# Chess ECO families
eco_family = {
    "A": "Flank Openings",
    "B": "Semi-Open (not French)",
    "C": "Open & French",
    "D": "Closed & Semi-Closed",
    "E": "Indian Defenses",
}

df["eco_letter"] = df["opening_eco"].str[0]

# 8-A  Number of games per ECO family
ax = axes[0]
eco_counts = df["eco_letter"].value_counts().reindex(["A","B","C","D","E"])
eco_names  = [eco_family.get(k, k) for k in eco_counts.index]
eco_colors = ['blue', 'purple', 'red', 'gray', 'green']
bars8 = ax.bar(eco_names, eco_counts.values,
               color=eco_colors, edgecolor='black', linewidth=1)
for bar in bars8:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 40,
            f"{int(height)}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(eco_names)))
ax.set_xticklabels(eco_names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Number of Games")
ax.set_title("Games per ECO Opening Family")

# 8-B  White win rate per ECO family
ax = axes[1]
eco_wr = []
for eco in ["A","B","C","D","E"]:
    sub = df[df["eco_letter"] == eco]
    n   = max(len(sub[sub["winner"].isin(["white","black","draw"])]), 1)
    eco_wr.append((sub["winner"] == "white").sum() / n * 100)

bars9 = ax.bar(eco_names, eco_wr,
               color=eco_colors, edgecolor='black', linewidth=1, alpha=0.8)
ax.axhline(50, color='gray', lw=1.2, ls="--", alpha=0.7, label="50% line")
for bar, val in zip(bars9, eco_wr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", color='black', fontsize=8, fontweight="bold")
ax.set_xticks(range(len(eco_names)))
ax.set_xticklabels(eco_names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("White Win Rate (%)")
ax.set_ylim(40, 60)
ax.legend(fontsize=8)
ax.set_title("Which Opening Family Favours White More?")

plt.tight_layout()
plt.savefig("fig8_eco_codes.png")
plt.close()
print("Saved fig8_eco_codes.png")


# ============================================================
# PRINT SUMMARY STATS
# ============================================================
print("\n" + "="*55)
print("  CHESS EDA — KEY FINDINGS SUMMARY")
print("="*55)
print(f"  Total games analysed   : {total:,}")
print(f"  White win rate         : {(df['winner']=='white').mean()*100:.2f}%")
print(f"  Black win rate         : {(df['winner']=='black').mean()*100:.2f}%")
print(f"  Draw rate              : {(df['winner']=='draw').mean()*100:.2f}%")
print(f"  Unique openings        : {df['opening_name'].nunique():,}")
print(f"  Avg game length        : {df['turns'].mean():.1f} turns")
print(f"  Median game length     : {df['turns'].median():.0f} turns")
print(f"  Shortest game          : {df['turns'].min()} turns")
print(f"  Longest game           : {df['turns'].max()} turns")
print(f"  Most common ending     : {df['victory_status'].value_counts().index[0]}")
print(f"  Most popular opening   : {df['opening_name'].value_counts().index[0]}")
print("="*55)
print("\nAll 8 figures saved successfully!")
