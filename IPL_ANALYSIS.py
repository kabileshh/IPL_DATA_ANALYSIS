
import zipfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

zip_path = 'ipl_json.zip'
zip_file = zipfile.ZipFile(zip_path, 'r')
json_files = [name for name in zip_file.namelist() if name.endswith('.json')]

match_rows = []
ball_rows = []
for json_name in tqdm(json_files):
    with zip_file.open(json_name) as file_obj:
        match_obj = json.load(file_obj)
    info_obj = match_obj.get('info', {})
    innings_list = match_obj.get('innings', [])
    match_id_val = json_name.replace('.json', '')
    dates_list = info_obj.get('dates', [])
    match_date = dates_list[0] if len(dates_list) > 0 else None
    teams_list = info_obj.get('teams', [])
    toss_obj = info_obj.get('toss', {})
    outcome_obj = info_obj.get('outcome', {})
    winner_team = outcome_obj.get('winner')
    if winner_team is None and outcome_obj.get('result') == 'tie':
        winner_team = outcome_obj.get('eliminator')
    match_rows.append({
        'match_id': match_id_val,
        'season': info_obj.get('season'),
        'date': match_date,
        'city': info_obj.get('city'),
        'venue': info_obj.get('venue'),
        'team1': teams_list[0] if len(teams_list) > 0 else None,
        'team2': teams_list[1] if len(teams_list) > 1 else None,
        'toss_winner': toss_obj.get('winner'),
        'toss_decision': toss_obj.get('decision'),
        'match_winner': winner_team,
        'result': outcome_obj.get('result'),
        'by_runs': outcome_obj.get('by', {}).get('runs'),
        'by_wickets': outcome_obj.get('by', {}).get('wickets')
    })
    for innings_index, innings_obj in enumerate(innings_list, start=1):
        batting_team = innings_obj.get('team')
        for over_obj in innings_obj.get('overs', []):
            over_number = over_obj.get('over', 0)
            for delivery_obj in over_obj.get('deliveries', []):
                wicket_list = delivery_obj.get('wickets', [])
                ball_rows.append({
                    'match_id': match_id_val,
                    'season': info_obj.get('season'),
                    'date': match_date,
                    'innings': innings_index,
                    'batting_team': batting_team,
                    'over': over_number + 1,
                    'batter': delivery_obj.get('batter'),
                    'bowler': delivery_obj.get('bowler'),
                    'non_striker': delivery_obj.get('non_striker'),
                    'runs_batter': delivery_obj.get('runs', {}).get('batter', 0),
                    'runs_extras': delivery_obj.get('runs', {}).get('extras', 0),
                    'runs_total': delivery_obj.get('runs', {}).get('total', 0),
                    'extras_type': ','.join(list(delivery_obj.get('extras', {}).keys())) if delivery_obj.get('extras') else None,
                    'wicket_flag': 1 if len(wicket_list) > 0 else 0,
                    'wicket_kind': wicket_list[0].get('kind') if len(wicket_list) > 0 else None,
                    'player_out': wicket_list[0].get('player_out') if len(wicket_list) > 0 else None,
                    'fielder': wicket_list[0].get('fielders', [{}])[0].get('name') if len(wicket_list) > 0 and len(wicket_list[0].get('fielders', [])) > 0 else None
                })

matches_df = pd.DataFrame(match_rows)
balls_df = pd.DataFrame(ball_rows)

matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
matches_df['season'] = pd.to_numeric(matches_df['season'], errors='coerce').astype('Int64')
balls_df['date'] = pd.to_datetime(balls_df['date'], errors='coerce')
balls_df['season'] = pd.to_numeric(balls_df['season'], errors='coerce').astype('Int64')
balls_df['over'] = pd.to_numeric(balls_df['over'], errors='coerce').astype('Int64')
balls_df['runs_batter'] = pd.to_numeric(balls_df['runs_batter'], errors='coerce').fillna(0).astype(int)
balls_df['runs_extras'] = pd.to_numeric(balls_df['runs_extras'], errors='coerce').fillna(0).astype(int)
balls_df['runs_total'] = pd.to_numeric(balls_df['runs_total'], errors='coerce').fillna(0).astype(int)
balls_df['wicket_flag'] = pd.to_numeric(balls_df['wicket_flag'], errors='coerce').fillna(0).astype(int)
balls_df['ball_seq'] = balls_df.groupby(['match_id', 'innings', 'over']).cumcount() + 1
balls_df['phase'] = np.select(
    [balls_df['over'].between(1, 6), balls_df['over'].between(7, 15), balls_df['over'].between(16, 20)],
    ['Powerplay', 'Middle Overs', 'Death Overs'],
    default='Other'
)

cleaned_df = balls_df.merge(
    matches_df[['match_id', 'team1', 'team2', 'toss_winner', 'toss_decision', 'match_winner', 'venue', 'city', 'result']],
    on='match_id',
    how='left'
)

team_name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants'
}
for col_name in ['batting_team', 'team1', 'team2', 'toss_winner', 'match_winner']:
    cleaned_df[col_name] = cleaned_df[col_name].replace(team_name_map)
    if col_name in matches_df.columns:
        matches_df[col_name] = matches_df[col_name].replace(team_name_map)

cleaned_df['city'] = cleaned_df['city'].fillna('Unknown')
cleaned_df['result'] = cleaned_df['result'].fillna('normal')
cleaned_df['winning_batting_team_flag'] = (cleaned_df['batting_team'] == cleaned_df['match_winner']).astype(int)
cleaned_df.to_csv('ipl_cleaned_ball_by_ball.csv', index=False)
matches_df.to_csv('ipl_matches_summary.csv', index=False)

valid_matches_df = matches_df[matches_df['match_winner'].notna()].copy()
valid_matches_df['toss_winner_won'] = (valid_matches_df['toss_winner'] == valid_matches_df['match_winner']).astype(int)
toss_summary_df = pd.DataFrame({
    'Outcome': ['Toss Winner Won Match', 'Toss Loser Won Match'],
    'Matches': [valid_matches_df['toss_winner_won'].sum(), len(valid_matches_df) - valid_matches_df['toss_winner_won'].sum()]
})
toss_summary_df['Win Percentage'] = toss_summary_df['Matches'] / toss_summary_df['Matches'].sum() * 100

phase_team_df = cleaned_df.groupby(['match_id', 'batting_team', 'phase'], as_index=False)['runs_total'].sum()
phase_team_df = phase_team_df.merge(valid_matches_df[['match_id', 'match_winner']], on='match_id', how='inner')
phase_team_df['team_result'] = np.where(phase_team_df['batting_team'] == phase_team_df['match_winner'], 'Winning Teams', 'Losing Teams')
phase_avg_df = phase_team_df.groupby(['phase', 'team_result'], as_index=False)['runs_total'].mean()

recent_df = cleaned_df[cleaned_df['season'].between(2020, 2024, inclusive='both')].copy()
batting_2020_2024_df = recent_df.groupby('batter', as_index=False)['runs_batter'].sum().sort_values('runs_batter', ascending=False).head(5)

wicket_exclusions = ['run out', 'retired hurt', 'retired out', 'obstructing the field']
wickets_recent_df = recent_df[(recent_df['wicket_flag'] == 1) & (~recent_df['wicket_kind'].isin(wicket_exclusions))]
bowling_2020_2024_df = wickets_recent_df.groupby('bowler', as_index=False)['wicket_flag'].sum().sort_values('wicket_flag', ascending=False).head(5)

sns.set_theme(style='whitegrid')
plt.figure(figsize=(8, 5))
sns.barplot(data=toss_summary_df, x='Outcome', y='Win Percentage', palette=['#1f77b4', '#ff7f0e'])
plt.title('IPL Toss Impact on Match Results')
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.barplot(data=phase_avg_df[phase_avg_df['phase'] != 'Other'], x='phase', y='runs_total', hue='team_result', palette=['#d62728', '#2ca02c'])
plt.title('Average Runs by Match Phase: Winning vs Losing Teams')
plt.tight_layout()
plt.show()
