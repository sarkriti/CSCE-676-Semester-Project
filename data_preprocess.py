# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

CSV_FILE = "odp_idb_2001-2023_ddg_clean.csv"
OUTPUT_FILE = "output/gender_disparity_anomaly_results.csv"
FEATURE_FILE = "output/gender_disparity_dataset.csv"


def load_data(csv_file=CSV_FILE):
    df = pd.read_csv(csv_file)
    expected = {
        'disease', 'county', 'year', 'sex', 'cases', 'population',
        'rate', 'lower_95_ci', 'upper_95_ci', 'rate_per_100k_recalc'
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def filter_base_rows(df):
    df = df.copy()
    df = df[df['sex'].isin(['Male', 'Female'])]
    df = df[df['county'] != 'California']
    return df


def validate_unique_groups(df):
    dupes = (
        df.groupby(['disease', 'county', 'year', 'sex'])
          .size()
          .reset_index(name='n')
          .query('n > 1')
    )
    return dupes


def build_disparity_table(df):
    work = df.copy()

    work['derived_rate'] = np.where(
        work['population'] > 0,
        (work['cases'] / work['population']) * 100000,
        np.nan
    )

    work['analysis_rate'] = work['rate']
    work['analysis_rate'] = work['analysis_rate'].fillna(work['rate_per_100k_recalc'])
    work['analysis_rate'] = work['analysis_rate'].fillna(work['derived_rate'])

    agg = (
        work.groupby(['disease', 'county', 'year', 'sex'], as_index=False)
            .agg(
                cases=('cases', 'sum'),
                population=('population', 'sum'),
                analysis_rate=('analysis_rate', 'mean'),
                derived_rate=('derived_rate', 'mean')
            )
    )

    pivot = (
        agg.pivot(index=['disease', 'county', 'year'], columns='sex', values=['cases', 'population', 'analysis_rate'])
           .reset_index()
    )
    pivot.columns = ['_'.join(col).strip('_') for col in pivot.columns.to_flat_index()]

    rename_map = {
        'cases_Male': 'male_cases',
        'cases_Female': 'female_cases',
        'population_Male': 'male_population',
        'population_Female': 'female_population',
        'analysis_rate_Male': 'male_rate',
        'analysis_rate_Female': 'female_rate'
    }
    pivot = pivot.rename(columns=rename_map)

    for col in ['male_cases', 'female_cases', 'male_population', 'female_population', 'male_rate', 'female_rate']:
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot['total_cases'] = pivot[['male_cases', 'female_cases']].sum(axis=1, min_count=1)
    pivot['total_county_population'] = pivot[['male_population', 'female_population']].sum(axis=1, min_count=1)
    pivot['rate_diff'] = pivot['male_rate'] - pivot['female_rate']
    pivot['abs_rate_diff'] = pivot['rate_diff'].abs()

    eps = 1e-6
    pivot['rate_ratio'] = np.where(
        pivot['female_rate'].fillna(0) > 0,
        pivot['male_rate'] / pivot['female_rate'],
        np.nan
    )
    pivot['log_rate_ratio'] = np.where(
        pivot[['male_rate', 'female_rate']].notna().all(axis=1),
        np.log((pivot['male_rate'].fillna(0) + eps) / (pivot['female_rate'].fillna(0) + eps)),
        np.nan
    )

    pivot['both_zero_cases'] = ((pivot['male_cases'].fillna(0) == 0) & (pivot['female_cases'].fillna(0) == 0)).astype(int)
    pivot['only_male_cases'] = ((pivot['male_cases'].fillna(0) > 0) & (pivot['female_cases'].fillna(0) == 0)).astype(int)
    pivot['only_female_cases'] = ((pivot['female_cases'].fillna(0) > 0) & (pivot['male_cases'].fillna(0) == 0)).astype(int)
    pivot['small_case_count'] = (pivot['total_cases'].fillna(0) < 5).astype(int)
    pivot['log_total_cases'] = np.log1p(pivot['total_cases'].fillna(0))
    pivot['log_total_population'] = np.log1p(pivot['total_county_population'].fillna(0))

    return pivot


def score_anomalies_by_disease(disparity_df, contamination=0.03, min_rows=20):
    feature_cols = [
        'male_rate', 'female_rate', 'rate_diff', 'abs_rate_diff',
        'log_rate_ratio', 'log_total_cases', 'log_total_population'
    ]

    all_results = []
    for disease, grp in disparity_df.groupby('disease', dropna=False):
        grp = grp.copy()
        eligible = grp[(grp['both_zero_cases'] == 0)].copy()
        eligible = eligible.dropna(subset=feature_cols)

        grp['anomaly_score'] = np.nan
        grp['anomaly_flag'] = np.nan
        grp['model_rows_used'] = 0

        if len(eligible) >= min_rows:
            scaler = StandardScaler()
            X = scaler.fit_transform(eligible[feature_cols])

            model = IsolationForest(
                n_estimators=300,
                contamination=contamination,
                random_state=42
            )
            model.fit(X)

            scores = -model.score_samples(X)
            flags = model.predict(X)
            flags = np.where(flags == -1, 1, 0)

            grp.loc[eligible.index, 'anomaly_score'] = scores
            grp.loc[eligible.index, 'anomaly_flag'] = flags
            grp['model_rows_used'] = len(eligible)

        all_results.append(grp)

    result = pd.concat(all_results, ignore_index=True)
    result = result.sort_values(['disease', 'anomaly_flag', 'anomaly_score'], ascending=[True, False, False])
    return result


def main():
    df = load_data()
    df = filter_base_rows(df)

    dupes = validate_unique_groups(df)
    if not dupes.empty:
        print(f"Warning: found {len(dupes)} duplicated disease/county/year/sex groups. Script aggregates them before modeling.")
    else:
        print("No duplicated disease/county/year/sex groups found.")

    disparity_df = build_disparity_table(df)
    disparity_df.to_csv(FEATURE_FILE, index=False)
    print(f"Saved engineered disparity dataset to {FEATURE_FILE}")

    results = score_anomalies_by_disease(disparity_df)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved anomaly-scored dataset to {OUTPUT_FILE}")
    print(f"Rows in output: {len(results)}")
    print(f"Diseases in output: {results['disease'].nunique()}")


if __name__ == '__main__':
    main()
