# =====================================================
# CHESS GAMES EDA
# Part 1
# Load Dataset + Cleaning + Figure 1 + Figure 2
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

df = pd.read_csv("games.csv")

print("Shape of Dataset:", df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

print("\nWinner Counts")
print(df["winner"].value_counts())

print("\nVictory Status Counts")
print(df["victory_status"].value_counts())

# -----------------------------------------------------
# CREATE NEW COLUMNS
# -----------------------------------------------------

df["avg_rating"] = (df["white_rating"] + df["black_rating"]) / 2

df["rating_diff"] = df["white_rating"] - df["black_rating"]

df["abs_diff"] = abs(df["rating_diff"])

rating_bins = [0,1000,1200,1400,1600,1800,2000,3000]

rating_labels = [
    "<1000",
    "1000-1200",
    "1200-1400",
    "1400-1600",
    "1600-1800",
    "1800-2000",
    "2000+"
]

df["rating_group"] = pd.cut(
    df["avg_rating"],
    bins=rating_bins,
    labels=rating_labels
)

print("\nTotal Games:", len(df))

print("White Wins :", (df["winner"]=="white").sum())

print("Black Wins :", (df["winner"]=="black").sum())

print("Draws :", (df["winner"]=="draw").sum())


# =====================================================
# FIGURE 1
# WHO WINS?
# =====================================================

plt.figure(figsize=(15,5))

# -----------------------------------------
# Pie Chart
# -----------------------------------------

plt.subplot(1,3,1)

winner_counts = df["winner"].value_counts()

plt.pie(
    winner_counts,
    labels=winner_counts.index,
    autopct="%1.1f%%"
)

plt.title("Winner Distribution")

# -----------------------------------------
# Bar Chart
# -----------------------------------------

plt.subplot(1,3,2)

winner_counts.plot(kind="bar")

plt.title("Winner Distribution")
plt.xlabel("Winner")
plt.ylabel("Games")

# -----------------------------------------
# Victory Status
# -----------------------------------------

plt.subplot(1,3,3)

df["victory_status"].value_counts().plot(kind="bar")

plt.title("Victory Status")
plt.xlabel("Status")
plt.ylabel("Games")

plt.tight_layout()

plt.show()


# =====================================================
# FIGURE 2
# DOES WHITE HAVE AN ADVANTAGE?
# =====================================================

plt.figure(figsize=(14,10))

# -----------------------------------------
# White Win Rate by Rating Group
# -----------------------------------------

plt.subplot(2,2,1)

white_rate = []

for group in rating_labels:

    temp = df[df["rating_group"]==group]

    if len(temp)>0:

        rate = (temp["winner"]=="white").mean()*100

    else:

        rate = np.nan

    white_rate.append(rate)

plt.plot(rating_labels, white_rate, marker="o")

plt.title("White Win Rate by Rating")
plt.xlabel("Rating Group")
plt.ylabel("White Win %")

plt.xticks(rotation=30)

plt.grid(True)


# -----------------------------------------
# White Black Draw Percentage
# -----------------------------------------

plt.subplot(2,2,2)

white_percent=[]
black_percent=[]
draw_percent=[]

for group in rating_labels:

    temp=df[df["rating_group"]==group]

    if len(temp)>0:

        white_percent.append((temp["winner"]=="white").mean()*100)

        black_percent.append((temp["winner"]=="black").mean()*100)

        draw_percent.append((temp["winner"]=="draw").mean()*100)

    else:

        white_percent.append(0)

        black_percent.append(0)

        draw_percent.append(0)

x=np.arange(len(rating_labels))

plt.bar(x,white_percent,label="White")

plt.bar(x,black_percent,bottom=white_percent,label="Black")

bottom=[]

for i in range(len(white_percent)):
    bottom.append(white_percent[i]+black_percent[i])

plt.bar(x,draw_percent,bottom=bottom,label="Draw")

plt.xticks(x,rating_labels,rotation=30)

plt.ylabel("Percentage")

plt.title("Game Result by Rating Group")

plt.legend()


# -----------------------------------------
# Draw Rate
# -----------------------------------------

plt.subplot(2,2,3)

plt.bar(rating_labels,draw_percent)

plt.title("Draw Rate by Rating")

plt.xlabel("Rating Group")

plt.ylabel("Draw %")

plt.xticks(rotation=30)


# -----------------------------------------
# Rated vs Unrated
# -----------------------------------------

plt.subplot(2,2,4)

rated=df[df["rated"]==True]

unrated=df[df["rated"]==False]

rated_result=[
    (rated["winner"]=="white").mean()*100,
    (rated["winner"]=="black").mean()*100,
    (rated["winner"]=="draw").mean()*100
]

unrated_result=[
    (unrated["winner"]=="white").mean()*100,
    (unrated["winner"]=="black").mean()*100,
    (unrated["winner"]=="draw").mean()*100
]

x=np.arange(3)

width=0.35

plt.bar(x-width/2,rated_result,width,label="Rated")

plt.bar(x+width/2,unrated_result,width,label="Unrated")

plt.xticks(x,["White","Black","Draw"])

plt.ylabel("Percentage")

plt.title("Rated vs Unrated Games")

plt.legend()

plt.tight_layout()

plt.show()
# =====================================================
# FIGURE 3
# DOES RATING MATTER?
# =====================================================

plt.figure(figsize=(14,10))

# -------------------------------------------------
# Higher Rated Player Win Rate
# -------------------------------------------------

games = df[df["winner"].isin(["white","black"])].copy()

games["higher_rated_wins"] = (
    ((games["white_rating"] > games["black_rating"]) & (games["winner"]=="white")) |
    ((games["black_rating"] > games["white_rating"]) & (games["winner"]=="black"))
)

gap_bins = [0,50,100,200,300,500,1500]

gap_labels = [
    "0-50",
    "50-100",
    "100-200",
    "200-300",
    "300-500",
    "500+"
]

games["gap_group"] = pd.cut(
    games["abs_diff"],
    bins=gap_bins,
    labels=gap_labels
)

win_rate = games.groupby("gap_group")["higher_rated_wins"].mean()*100

plt.subplot(2,2,1)

plt.plot(gap_labels, win_rate, marker="o")

plt.title("Higher Rated Player Win Rate")
plt.xlabel("Rating Difference")
plt.ylabel("Win %")

plt.grid(True)



# -------------------------------------------------
# Rating Difference Distribution
# -------------------------------------------------

plt.subplot(2,2,2)

plt.hist(df["rating_diff"], bins=40)

plt.title("Rating Difference Distribution")
plt.xlabel("White Rating - Black Rating")
plt.ylabel("Games")



# -------------------------------------------------
# Winner Rating vs Loser Rating
# -------------------------------------------------

plt.subplot(2,2,3)

winner_rating = []

loser_rating = []

for i,row in games.iterrows():

    if row["winner"]=="white":

        winner_rating.append(row["white_rating"])
        loser_rating.append(row["black_rating"])

    else:

        winner_rating.append(row["black_rating"])
        loser_rating.append(row["white_rating"])

plt.boxplot([winner_rating,loser_rating],
            labels=["Winner","Loser"])

plt.title("Winner Rating vs Loser Rating")

plt.ylabel("Rating")



# -------------------------------------------------
# White Rating vs Black Rating
# -------------------------------------------------

plt.subplot(2,2,4)

plt.hist(df["white_rating"],
         bins=30,
         alpha=0.5,
         label="White")

plt.hist(df["black_rating"],
         bins=30,
         alpha=0.5,
         label="Black")

plt.title("White vs Black Rating")

plt.xlabel("Rating")

plt.ylabel("Games")

plt.legend()

plt.tight_layout()

plt.show()



# =====================================================
# FIGURE 4
# OPENING ANALYSIS
# =====================================================

plt.figure(figsize=(14,10))


# -------------------------------------------------
# Top 20 Openings
# -------------------------------------------------

plt.subplot(2,2,1)

top_openings = df["opening_name"].value_counts().head(20)

top_openings.plot(kind="barh")

plt.title("Top 20 Most Played Openings")

plt.xlabel("Games")



# -------------------------------------------------
# Best Openings for White
# -------------------------------------------------

plt.subplot(2,2,2)

opening_stats = df.groupby("opening_name").agg(

    games=("winner","count"),

    white_wins=("winner",
                lambda x:(x=="white").sum())

)

opening_stats = opening_stats[opening_stats["games"]>=100]

opening_stats["white_win_rate"] = (

    opening_stats["white_wins"] /
    opening_stats["games"]

)*100

best_openings = opening_stats.sort_values(
    "white_win_rate",
    ascending=False
).head(15)

plt.barh(best_openings.index,
         best_openings["white_win_rate"])

plt.title("Top Openings for White")

plt.xlabel("White Win %")



# -------------------------------------------------
# Popularity vs White Win Rate
# -------------------------------------------------

plt.subplot(2,2,3)

plt.scatter(

    opening_stats["games"],

    opening_stats["white_win_rate"]

)

plt.title("Popularity vs White Win Rate")

plt.xlabel("Number of Games")

plt.ylabel("White Win %")



# -------------------------------------------------
# Opening Diversity
# -------------------------------------------------

plt.subplot(2,2,4)

opening_count = df["opening_name"].value_counts()

coverage = opening_count.cumsum()/opening_count.sum()*100

plt.plot(range(1,101),

         coverage.iloc[:100])

plt.title("Opening Diversity")

plt.xlabel("Top N Openings")

plt.ylabel("Coverage %")

plt.grid(True)

plt.tight_layout()

plt.show()
# =====================================================
# FIGURE 5
# GAME LENGTH ANALYSIS
# =====================================================

plt.figure(figsize=(14,10))

# -------------------------------------------------
# Distribution of Game Length
# -------------------------------------------------

plt.subplot(2,2,1)

plt.hist(df["turns"], bins=40)

plt.title("Distribution of Game Length")
plt.xlabel("Number of Turns")
plt.ylabel("Games")

print("Average Turns :", df["turns"].mean())
print("Median Turns :", df["turns"].median())


# -------------------------------------------------
# Game Length by Winner
# -------------------------------------------------

plt.subplot(2,2,2)

white_turns = df[df["winner"]=="white"]["turns"]

black_turns = df[df["winner"]=="black"]["turns"]

draw_turns = df[df["winner"]=="draw"]["turns"]

plt.boxplot(
    [white_turns, black_turns, draw_turns],
    labels=["White","Black","Draw"]
)

plt.title("Game Length by Winner")
plt.ylabel("Turns")


# -------------------------------------------------
# White Win Rate by Game Length
# -------------------------------------------------

plt.subplot(2,2,3)

turn_bins = [0,20,40,60,80,120,400]

turn_labels = [
    "0-20",
    "20-40",
    "40-60",
    "60-80",
    "80-120",
    "120+"
]

df["turn_group"] = pd.cut(
    df["turns"],
    bins=turn_bins,
    labels=turn_labels
)

white_rate = []

for group in turn_labels:

    temp = df[df["turn_group"]==group]

    if len(temp)>0:

        rate = (temp["winner"]=="white").mean()*100

    else:

        rate = 0

    white_rate.append(rate)

plt.plot(turn_labels, white_rate, marker="o")

plt.title("White Win Rate by Game Length")
plt.xlabel("Turns")
plt.ylabel("White Win %")

plt.grid(True)


# -------------------------------------------------
# Average Turns by Victory Status
# -------------------------------------------------

plt.subplot(2,2,4)

avg_turns = df.groupby("victory_status")["turns"].mean()

avg_turns.plot(kind="bar")

plt.title("Average Turns by Victory Status")

plt.xlabel("Victory Status")

plt.ylabel("Average Turns")

plt.tight_layout()

plt.show()



# =====================================================
# FIGURE 6
# BEGINNER VS EXPERT
# =====================================================

plt.figure(figsize=(15,10))


# -------------------------------------------------
# Skill Groups
# -------------------------------------------------

skill_bins = [0,1200,1600,2000,3000]

skill_labels = [
    "<1200",
    "1200-1600",
    "1600-2000",
    "2000+"
]

df["skill_group"] = pd.cut(
    df["avg_rating"],
    bins=skill_bins,
    labels=skill_labels
)


# -------------------------------------------------
# Average Turns by Skill
# -------------------------------------------------

plt.subplot(2,3,1)

avg_turns = df.groupby("skill_group")["turns"].mean()

avg_turns.plot(kind="bar")

plt.title("Average Turns by Skill")

plt.xlabel("Skill Group")

plt.ylabel("Average Turns")


# -------------------------------------------------
# Resign vs Timeout
# -------------------------------------------------

plt.subplot(2,3,2)

resign = []

timeout = []

for group in skill_labels:

    temp = df[df["skill_group"]==group]

    resign.append(
        (temp["victory_status"]=="resign").mean()*100
    )

    timeout.append(
        (temp["victory_status"]=="outoftime").mean()*100
    )

x = np.arange(len(skill_labels))

width = 0.35

plt.bar(x-width/2,resign,width,label="Resign")

plt.bar(x+width/2,timeout,width,label="Timeout")

plt.xticks(x,skill_labels)

plt.ylabel("Percentage")

plt.title("Resign vs Timeout")

plt.legend()


# -------------------------------------------------
# Favourite Opening
# -------------------------------------------------

plt.subplot(2,3,3)

favorite_openings = []

opening_count = []

for group in skill_labels:

    temp = df[df["skill_group"]==group]

    top = temp["opening_name"].value_counts()

    favorite_openings.append(top.index[0])

    opening_count.append(top.iloc[0])

plt.bar(skill_labels,opening_count)

plt.title("Most Used Opening")

plt.xlabel("Skill Group")

plt.ylabel("Games")

print("\nFavourite Opening by Skill")

for i in range(len(skill_labels)):

    print(skill_labels[i],":",favorite_openings[i])


# -------------------------------------------------
# Rating Distribution
# -------------------------------------------------

plt.subplot(2,3,4)

for group in skill_labels:

    temp = df[df["skill_group"]==group]

    plt.hist(
        temp["avg_rating"],
        bins=20,
        alpha=0.5,
        label=group
    )

plt.title("Rating Distribution")

plt.xlabel("Average Rating")

plt.ylabel("Games")

plt.legend()


# -------------------------------------------------
# Victory Status by Skill
# -------------------------------------------------

plt.subplot(2,3,5)

status = (
    pd.crosstab(
        df["skill_group"],
        df["victory_status"],
        normalize="index"
    )
    *100
)

status.plot(
    kind="bar",
    stacked=True,
    ax=plt.gca()
)

plt.title("Victory Status by Skill")

plt.xlabel("Skill Group")

plt.ylabel("Percentage")

plt.legend(fontsize=7)


# -------------------------------------------------
# Number of Games
# -------------------------------------------------

plt.subplot(2,3,6)

df["skill_group"].value_counts().sort_index().plot(kind="bar")

plt.title("Games in Each Skill Group")

plt.xlabel("Skill Group")

plt.ylabel("Games")

plt.tight_layout()

plt.show()
# =====================================================
# FIGURE 7
# INTERESTING QUESTIONS
# =====================================================

plt.figure(figsize=(15,10))

# -------------------------------------------------
# Opening with Highest Draw Rate
# -------------------------------------------------

plt.subplot(2,3,1)

opening_draw = df.groupby("opening_name").agg(
    games=("winner","count"),
    draws=("winner",lambda x:(x=="draw").sum())
)

opening_draw = opening_draw[opening_draw["games"]>=80]

opening_draw["draw_rate"] = (
    opening_draw["draws"] /
    opening_draw["games"]
)*100

top_draw = opening_draw.sort_values(
    "draw_rate",
    ascending=False
).head(10)

plt.barh(top_draw.index, top_draw["draw_rate"])

plt.title("Highest Draw Rate Openings")
plt.xlabel("Draw %")


# -------------------------------------------------
# Checkmate Game Length
# -------------------------------------------------

plt.subplot(2,3,2)

mate_games = df[df["victory_status"]=="mate"]

plt.hist(mate_games["turns"], bins=30)

plt.title("Turns Before Checkmate")

plt.xlabel("Turns")

plt.ylabel("Games")


# -------------------------------------------------
# White Win Rate in Popular Openings
# -------------------------------------------------

plt.subplot(2,3,3)

popular = df["opening_name"].value_counts().head(15).index

white_rate = []

for opening in popular:

    temp = df[df["opening_name"]==opening]

    rate = (temp["winner"]=="white").mean()*100

    white_rate.append(rate)

plt.barh(popular,white_rate)

plt.title("White Win Rate in Popular Openings")

plt.xlabel("White Win %")


# -------------------------------------------------
# Very Short Games
# -------------------------------------------------

plt.subplot(2,3,4)

short_games = df[df["turns"]<=15]

short_games["winner"].value_counts().plot(kind="bar")

plt.title("Winner in Very Short Games")

plt.xlabel("Winner")

plt.ylabel("Games")


# -------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------

plt.subplot(2,3,5)

numeric = df[
    [
        "white_rating",
        "black_rating",
        "avg_rating",
        "turns",
        "abs_diff"
    ]
]

corr = numeric.corr()

plt.imshow(corr)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Matrix")


# -------------------------------------------------
# Opening Length
# -------------------------------------------------

plt.subplot(2,3,6)

opening_bins=[0,2,4,6,8,10,30]

opening_labels=[
    "1-2",
    "3-4",
    "5-6",
    "7-8",
    "9-10",
    "11+"
]

df["opening_group"]=pd.cut(
    df["opening_ply"],
    bins=opening_bins,
    labels=opening_labels
)

opening_rate=[]

for group in opening_labels:

    temp=df[df["opening_group"]==group]

    if len(temp)>0:

        rate=(temp["winner"]=="white").mean()*100

    else:

        rate=0

    opening_rate.append(rate)

plt.bar(opening_labels,opening_rate)

plt.title("Opening Length vs White Win")

plt.xlabel("Opening Length")

plt.ylabel("White Win %")

plt.tight_layout()

plt.show()



# =====================================================
# FIGURE 8
# ECO ANALYSIS
# =====================================================

plt.figure(figsize=(12,5))

eco_names={
    "A":"Flank",
    "B":"Semi-Open",
    "C":"Open",
    "D":"Closed",
    "E":"Indian"
}

df["eco_group"]=df["opening_eco"].str[0]


# -------------------------------------------------
# Games by ECO
# -------------------------------------------------

plt.subplot(1,2,1)

eco_count=df["eco_group"].value_counts().sort_index()

plt.bar(
    eco_count.index,
    eco_count.values
)

plt.xticks(
    eco_count.index,
    [eco_names[i] for i in eco_count.index],
    rotation=20
)

plt.title("Games by ECO")

plt.xlabel("Opening Family")

plt.ylabel("Games")


# -------------------------------------------------
# White Win Rate by ECO
# -------------------------------------------------

plt.subplot(1,2,2)

eco_rate=[]

labels=[]

for eco in ["A","B","C","D","E"]:

    temp=df[df["eco_group"]==eco]

    labels.append(eco_names[eco])

    eco_rate.append(
        (temp["winner"]=="white").mean()*100
    )

plt.bar(labels,eco_rate)

plt.title("White Win Rate by ECO")

plt.xlabel("Opening Family")

plt.ylabel("White Win %")

plt.tight_layout()

plt.show()



# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n==============================")
print("CHESS DATASET SUMMARY")
print("==============================")

print("Total Games :",len(df))

print("White Wins :",
      (df["winner"]=="white").sum())

print("Black Wins :",
      (df["winner"]=="black").sum())

print("Draws :",
      (df["winner"]=="draw").sum())

print()

print("White Win Rate :",
      round((df["winner"]=="white").mean()*100,2),
      "%")

print("Black Win Rate :",
      round((df["winner"]=="black").mean()*100,2),
      "%")

print("Draw Rate :",
      round((df["winner"]=="draw").mean()*100,2),
      "%")

print()

print("Average Turns :",
      round(df["turns"].mean(),2))

print("Median Turns :",
      df["turns"].median())

print("Shortest Game :",
      df["turns"].min())

print("Longest Game :",
      df["turns"].max())

print()

print("Unique Openings :",
      df["opening_name"].nunique())

print("Most Popular Opening :",
      df["opening_name"].value_counts().idxmax())

print("Most Common Ending :",
      df["victory_status"].value_counts().idxmax())

print("==============================")
