import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load data from csv
df = pd.read_csv('file.csv')

# Filter data for year 2018
df = df[df['year'] == 2018]
df['xGA'] = df['xGA'].round()

# Group data by league and team
grouped = df.groupby(['league', 'team'])

models_lr = {}
train_X_adaboost = []
train_y_adaboost = []

# Iterate over groups
for (league, team), group in grouped:
    # Train a linear regression model
    model_lr = LinearRegression()
    model_lr.fit(group[['xG']], group['scored'])

    # Store the model in the dictionary
    models_lr[(league, team)] = model_lr

    # Create the training dataset for AdaBoost
    for _, row in group.iterrows():
        xG, xGA, scored, result = row['xG'], row['xGA'], row['scored'], row['result']

        # Check the conditions for each result and scored value
        if (scored > xGA) and (result == 'w'):
            train_y_adaboost.append('win')
        elif (scored < xGA) and (result == 'w'):
            train_y_adaboost.append('win')
        elif (scored == xGA) and (result == 'w'):
            train_y_adaboost.append('win')
        elif (scored > xGA) and (result == 'l'):
            train_y_adaboost.append('lose')
        elif (scored < xGA) and (result == 'l'):
            train_y_adaboost.append('lose')
        elif (scored == xGA) and (result == 'l'):
            train_y_adaboost.append('lose')
        elif (scored > xGA) and (result == 'd'):
            train_y_adaboost.append('draw')
        elif (scored < xGA) and (result == 'd'):
            train_y_adaboost.append('draw')
        elif (scored == xGA) and (result == 'd'):
            train_y_adaboost.append('draw')

        train_X_adaboost.append([xG, xGA])

# Train AdaBoost classifier
model_adaboost = AdaBoostClassifier(random_state=0, n_estimators=100)
model_adaboost.fit(train_X_adaboost, train_y_adaboost)

# Prepare data for 2019
df_2019 = pd.read_csv('file.csv')
df_2019 = df_2019[df_2019['year'] == 2019]
df_2019['xGA'] = df_2019['xGA'].round()

# Initialize an empty DataFrame for the 2019 standings
standings_2019 = pd.DataFrame(columns=['league', 'team', 'points'])
standings_2019['points'] = standings_2019['points'].astype(int)

# Iterate over each team in the 2019 DataFrame
for (league, team), group in df_2019.groupby(['league', 'team']):
    if (league, team) not in models_lr:
        print(f"No 2018 data for {team} in {league}. Skipping.")
        continue

    points = 0
    for idx, row in group.iterrows():
        xG, xGA = row['xG'], row['xGA']
        predicted_score = models_lr[(league, team)].predict([[xG]])[0]
        predicted_result = model_adaboost.predict([[xG, xGA]])[0]

        if predicted_result == 'win':
            points += 3
        elif predicted_result == 'draw':
            points += 1

    standings_2019 = standings_2019.append({'league': league, 'team': team, 'points': points}, ignore_index=True)

# For each league, sort teams by points and output to a CSV file
for league, data in standings_2019.groupby('league'):
    data = data.sort_values('points', ascending=False)
    data.to_csv(f'AdaBoost_{league}_standings.csv', index=False)

# Selected teams for graph
selected_teams = [('La_liga', 'Barcelona'), ('La_liga', 'Celta Vigo'), ('La_liga', 'Real Valladolid'),
                  ('EPL', 'Manchester City'), ('EPL', 'Liverpool'), ('EPL', 'Chelsea'), ('EPL', 'Manchester United')]

# Separate graphs for training and predictions for each selected team
for (league, team) in selected_teams:
    plt.figure(figsize=(12, 8))

    # Define custom colormap
    colors = ListedColormap(['r', 'g', 'b'])

    # Scatter plot of the training data points
    X_train = grouped.get_group((league, team))[['xG', 'xGA']]
    y_train = model_adaboost.predict(X_train)
    y_train_labels = np.unique(train_y_adaboost)
    y_train_numeric = np.array([np.where(y_train_labels == label)[0][0] for label in y_train])
    plt.scatter(X_train['xG'], X_train['xGA'], c=y_train_numeric, cmap=colors, label=f"{team} (Training)")

    plt.xlabel('xG')
    plt.ylabel('xGA')
    plt.title(f'AdaBoost Training for {team} ({league})')

    # Adding legend for colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=12)
                       for label, color in zip(y_train_labels, colors.colors)]
    plt.legend(handles=legend_elements)

    plt.show()

    # Scatter plot of the predicted data points for 2019
    plt.figure(figsize=(12, 8))

    X_predict = df_2019[(df_2019['league'] == league) & (df_2019['team'] == team)][['xG', 'xGA']]
    y_predict = model_adaboost.predict(X_predict)
    y_predict_labels = np.unique(y_predict)
    y_predict_numeric = np.array([np.where(y_train_labels == label)[0][0] for label in y_predict])
    plt.scatter(X_predict['xG'], X_predict['xGA'], c=y_predict_numeric, cmap=colors, marker='o', s=100, label=f"{team} (Prediction)")

    # Adding legend for colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8)
                       for label, color in zip(y_predict_labels, colors.colors)]
    plt.legend(handles=legend_elements)

    plt.xlabel('xG')
    plt.ylabel('xGA')
    plt.title(f'AdaBoost Prediction for {team} ({league})')
    plt.show()
