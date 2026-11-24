"""SIDPOL: funciones comunes de las Notebooks 02-04.

Las notebooks muestran la calibración, los cortes temporales, las comparaciones y la construcción de figuras.
Este módulo reúne rutas, controles, variables predictoras, modelos, estadística y exportación de tablas.

Controles: C1 avisa de diferencias de versiones; los controles de integridad detienen ante un error.
En modo reentrenar se exige además el entorno fijado antes de ajustar modelos:
  C1 versiones (aviso, no detiene)            C6 huella de los pronósticos en niveles v2
  C2 huella del panel y de las constantes Q_j C7 la evidencia v2 usa los hiperparámetros congelados
  C3 huella de la calibración en niveles v2   C8 corte temporal: cutoff = mes pronosticado − 1
  C4 predicción por etapas = modelo con n     C9 claves exactas, naïve y observados desde el panel
  C5 cadena 02 → 03 (configuración)           C10 cadena 03 → 04 (pronósticos, configuración, scores)
"""
import hashlib
import importlib.metadata as package_metadata
import json
import platform
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

TARGETS = ['ROBO', 'HURTO']
BREAK_DATE = pd.Timestamp('2024-11-01')            # cambio de registro de denuncias
SEED = 42
EXTERNAL_MONTHS = pd.date_range('2023-01-01', '2026-07-01', freq='MS')
CALIBRATION_MONTHS = pd.date_range('2022-01-01', '2022-12-01', freq='MS')
EXPECTED_ENV = {'numpy': '2.1.3', 'pandas': '2.2.3', 'xgboost': '3.4.1', 'scipy': '1.16.3',
                'scikit-learn': '1.6.1', 'matplotlib': '3.10.0'}
INPUT_HASHES = {   # datos auditados por la Notebook 01 que usan los modelos
    'sidpol_panel_lima_robo_hurto.csv': 'd0be7ea4da1ad68cf88b2b6bf3c0b76abc063b83c4897aa83746ec322885f868',
    'sidpol_mase_constants.csv': 'f4493cd68871d3c0d3eb62bf745280b72538ac928ad3d8c38cc0b895077ca896'}
EVIDENCE_HASHES = {   # corrida principal v2 (modelos en niveles), ya auditada
    'calibracion_niveles_original/notebook02_internal_predictions.csv':
        'e793e7e86009440da396ce5d8a6aa0c2ea91b2f5cbbc52038050bfc1371f52c4',
    'pronosticos_niveles_v2/notebook03_external_predictions.csv':
        'df26a2eae7254136075fce0f7503b113e1b686d1365932d9bf2a9827d493677e',
    'pronosticos_niveles_v2/notebook03_global42_predictions.csv':
        'f3318cf4411e06debea268b2c5df22ab5cc9bd29a769c5fbefcbf44a52812523'}
STRATEGIES = ['local_HL', 'global_HL', 'global_HG', 'global_42', 'naive1', 'naive_seasonal',
              'local_dz_HL', 'global_dz_HL', 'global_dz_HG', 'ridge_dz', 'ses', 'ses_pendiente']
LEVEL_H = {'local_HL': 'H_L', 'global_HL': 'H_L', 'global_HG': 'H_G', 'global_42': 'H_G'}
TIE_TOL = 1e-12        # |diferencia| menor: empate numérico (residuo de punto flotante)
N_BOOT = 5000


# ----------------------------------------------------------------------------- rutas, archivos y controles
def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def project(project_dir, results_dir="resultados"):
    """Rutas del proyecto a partir de la carpeta que contiene las notebooks."""
    root = Path(project_dir)
    res = root / results_dir
    require(res.resolve().is_relative_to(root.resolve()) and res.resolve() != root.resolve(),
            'CARPETA_RESULTADOS debe ser una subcarpeta del proyecto.')
    p = SimpleNamespace(root=root, audit=root / '01. Auditoria SIDPOL', evidence=root / 'evidencia_registrada',
                        cal=res / '02_calibracion', forecast=res / '03_pronosticos', analysis=res / '04_resultados')
    p.config = p.cal / 'configuracion_congelada.json'
    p.scores = p.cal / 'calibracion_scores.csv'
    p.forecasts = p.forecast / 'pronosticos_2023_2026.csv'
    require(p.audit.exists(), f'No encuentro "{p.audit.name}" en {root}. Revise PROJECT_DIR.')
    return p


def check_env():
    """C1: registra las versiones y avisa si difieren de las fijadas (no detiene: fijar versiones no garantiza bits)."""
    env = {'python': platform.python_version()}
    for name in EXPECTED_ENV:
        try:
            env[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            env[name] = None
    wrong = {k: (v, env[k]) for k, v in EXPECTED_ENV.items() if env[k] != v}
    if wrong:
        warnings.warn(f'Versiones distintas de requirements_modelado.txt (esperada, actual): {wrong}. '
                      'En modo revisar se verifican archivos guardados y se registra este entorno. '
                      'No se está reproduciendo el entrenamiento. Consulte README.md.')
    return env


def load_inputs(p):
    """C2: panel y constantes Q_j de la Notebook 01, verificados por huella."""
    for name, expected in INPUT_HASHES.items():
        require(sha256_file(p.audit / name) == expected,
                f'Cambió {name}. Restaure el archivo de la entrega; no ejecute la Notebook 01.')
    panel = pd.read_csv(p.audit / 'sidpol_panel_lima_robo_hurto.csv', dtype={'UBIGEO': 'string'}, parse_dates=['fecha'])
    q = pd.read_csv(p.audit / 'sidpol_mase_constants.csv', dtype={'UBIGEO': 'string'})
    panel = panel.sort_values(['target', 'UBIGEO', 'fecha']).reset_index(drop=True)
    Q = {(r.target, str(r.UBIGEO)): float(r.Q_j) for r in q.itertuples()}
    return panel, Q


def load_evidence(p, name):
    """C3/C6: un archivo de la corrida principal v2, verificado con una huella fija en el código."""
    path = p.evidence / name
    require(sha256_file(path) == EVIDENCE_HASHES[name], f'Cambió {name}. Restaure el archivo de la entrega.')
    return pd.read_csv(path, dtype={'UBIGEO': str}, float_precision='round_trip')


def save_manifest(folder, env, outputs, **extra):
    write_json(Path(folder) / 'manifest.json', {
        'status': 'COMPLETED', 'environment': env, 'input_hashes': INPUT_HASHES, **extra,
        'output_hashes': {str(Path(o).relative_to(folder)): sha256_file(o) for o in sorted(outputs)
                          if Path(o).resolve() != (Path(folder) / 'manifest.json').resolve()}})


def load_config(p):
    """C5: la configuración congelada debe ser la que dejó la Notebook 02."""
    man = json.loads((p.cal / 'manifest.json').read_text())
    require(sha256_file(p.config) == man['output_hashes'][p.config.name],
            'La configuración no coincide con su manifiesto. Restaure ambos archivos de la misma entrega; revisar no los regenera.')
    return json.loads(p.config.read_text())


def load_forecasts(p):
    """C10: pronósticos, configuración y scores deben venir de la misma cadena 02 → 03."""
    cal_man = json.loads((p.cal / 'manifest.json').read_text())
    fc_man = json.loads((p.forecast / 'manifest.json').read_text())
    require(sha256_file(p.forecasts) == fc_man['output_hashes'][p.forecasts.name]
            and fc_man['config_sha256'] == sha256_file(p.config) == cal_man['output_hashes'][p.config.name]
            and sha256_file(p.scores) == cal_man['output_hashes'][p.scores.name],
            'Pronósticos, configuración o scores no son de la misma corrida. Restaure la cadena 02 → 03 de la misma entrega; revisar no la regenera.')
    return pd.read_csv(p.forecasts, dtype={'UBIGEO': str}, parse_dates=['cutoff', 'forecast_month'],
                       float_precision='round_trip')


# ----------------------------------------------------------------------------- features
LEVEL_FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3', 'rolling_mean_6',
                  'mes_sin', 'mes_cos', 'UBIGEO']
REL_SOURCES = ['lag_2', 'lag_3', 'lag_6', 'lag_12', 'rolling_mean_3', 'rolling_mean_6']
DZ_FEATURES = ['lag_1'] + [f'r_{c}' for c in REL_SOURCES] + ['mes_sin', 'mes_cos', 'UBIGEO']
RIDGE_FEATURES = [f'r_{c}' for c in REL_SOURCES] + ['mes_sin', 'mes_cos']


def build_features(data, districts):
    """Rezagos y medias móviles de z = log1p(y), siempre con meses ANTERIORES de la misma serie."""
    d = data.sort_values(['target', 'UBIGEO', 'fecha']).copy()
    d['z'] = np.log1p(d.y.astype(float))
    grouped = d.groupby(['target', 'UBIGEO'], observed=True)['z']
    for lag in [1, 2, 3, 6, 12]:
        d[f'lag_{lag}'] = grouped.shift(lag)
    for window in [3, 6]:
        d[f'rolling_mean_{window}'] = grouped.transform(lambda s: s.shift(1).rolling(window).mean())
    month = d.fecha.dt.month
    d['mes_sin'], d['mes_cos'] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    d['UBIGEO'] = d.UBIGEO.astype(pd.CategoricalDtype(categories=districts, ordered=False))
    d = d.dropna(subset=['lag_12']).copy()                 # sin imputar
    d['dz'] = d.z - d.lag_1                                # Δz: cambio respecto del mes anterior
    for c in REL_SOURCES:
        d[f'r_{c}'] = d[c] - d.lag_1
    return d.reset_index(drop=True)


def to_counts_from_z(z_hat):
    return np.maximum(np.expm1(z_hat), 0.0)


def to_counts_from_dz(lag_1, dz_hat):
    return np.clip(np.expm1(np.asarray(lag_1, dtype=float) + np.asarray(dz_hat, dtype=float)), 0, None)


# ----------------------------------------------------------------------------- modelos
LEVEL_PARAMS = {'objective': 'reg:squarederror', 'tree_method': 'hist', 'enable_categorical': True,
                'random_state': 42, 'learning_rate': 0.05, 'min_child_weight': 5, 'subsample': 1.0,
                'colsample_bytree': 1.0, 'reg_lambda': 1.0, 'reg_alpha': 0.0, 'n_jobs': -1}
DZ_PARAMS = {**LEVEL_PARAMS, 'n_jobs': 1}   # un hilo; se registra el entorno, sin asumir identidad numérica entre entornos


def xgb(params, max_depth, n_estimators):
    from xgboost import XGBRegressor
    return XGBRegressor(**params, max_depth=int(max_depth), n_estimators=int(n_estimators))


def dz_forecast(params, train, test, h):
    model = xgb(params, h['max_depth'], h['n_estimators']).fit(train[DZ_FEATURES], train.dz)
    return to_counts_from_dz(test.lag_1, model.predict(test[DZ_FEATURES]))


def ridge_forecast(train, test, alpha):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=float(alpha)).fit(train[RIDGE_FEATURES], train.dz)
    return to_counts_from_dz(test.lag_1, model.predict(test[RIDGE_FEATURES]))


SES_ALPHAS = np.linspace(0.05, 1, 20)


def ses_level(x):
    """SES sobre la historia x (log1p): α de la grilla con menor error a un paso dentro de la historia."""
    best = None
    for alpha in SES_ALPHAS:
        level, sse = x[0], 0.0
        for value in x[1:]:
            sse += (value - level) ** 2
            level = level + alpha * (value - level)
        if best is None or sse < best[0]:
            best = (sse, level)
    return best[1]


def ses_slope_level(x, ses):
    """Referencia ad hoc: SES + ½ pendiente OLS. NO es el método Theta estándar."""
    return ses + 0.5 * np.polyfit(np.arange(len(x)), x, 1)[0]


def staged_check(features):
    """C4: predecir con los primeros n árboles de un modelo de 300 equivale a entrenar con n árboles."""
    tr = features[(features.target == 'ROBO') & (features.fecha < CALIBRATION_MONTHS[0])]
    te = features[(features.target == 'ROBO') & (features.fecha == CALIBRATION_MONTHS[0])]
    big = xgb(DZ_PARAMS, 2, 300).fit(tr[DZ_FEATURES], tr.dz)
    for n in [25, 100]:
        small = xgb(DZ_PARAMS, 2, n).fit(tr[DZ_FEATURES], tr.dz)
        require(np.abs(small.predict(te[DZ_FEATURES]) - big.predict(te[DZ_FEATURES], iteration_range=(0, n))).max()
                < 1e-6, 'C4: la predicción por etapas no equivale a entrenar cada tamaño.')


def calibration_scores(pred, keys, Q):
    """Score 2022 = mediana entre distritos del MASE del distrito (media de los 12 meses)."""
    pred = pred.copy()
    pred['Q_j'] = [Q[(t, u)] for t, u in zip(pred.target, pred.UBIGEO)]
    pred['scaled_error'] = (pred.y_true - pred.y_pred).abs() / pred.Q_j
    district = pred.groupby(keys + ['UBIGEO'], as_index=False, observed=True).agg(
        district_MASE=('scaled_error', 'mean'), n_origins=('origin', 'nunique'))
    scores = district.groupby(keys, as_index=False).agg(
        Score=('district_MASE', 'median'),
        IQR_district_MASE=('district_MASE', lambda s: s.quantile(.75) - s.quantile(.25)),
        n_districts=('UBIGEO', 'nunique'))
    return pred, scores


def level_evidence_forecasts(p, selected):
    """C6/C7: pronósticos en niveles de la corrida v2, con los hiperparámetros congelados."""
    parts = []
    for name in ['pronosticos_niveles_v2/notebook03_external_predictions.csv',
                 'pronosticos_niveles_v2/notebook03_global42_predictions.csv']:
        old = load_evidence(p, name)
        old = old[old.strategy.isin(LEVEL_H)]
        for (target, strategy), g in old.groupby(['target', 'strategy']):
            h = selected[target][LEVEL_H[strategy]]
            require(g.max_depth.astype(float).eq(h['max_depth']).all() and g.n_estimators.astype(float).eq(
                h['n_estimators']).all(), f'C7: {target}/{strategy} no usa los hiperparámetros congelados.')
        parts.append(old)
    out = pd.concat(parts, ignore_index=True)
    out['cutoff'], out['forecast_month'] = pd.to_datetime(out.cutoff), pd.to_datetime(out.forecast_month)
    return out


def check_forecasts(fc, panel, districts, excluded):
    """C9: cada estrategia tiene exactamente target × distrito × mes; observados y naïve salen del panel."""
    y = panel.assign(UBIGEO=panel.UBIGEO.astype(str)).set_index(['target', 'UBIGEO', 'fecha']).y.astype(float)
    look = lambda g, k: y.reindex(pd.MultiIndex.from_arrays(
        [g.target, g.UBIGEO, g.forecast_month - pd.DateOffset(months=k)])).to_numpy()
    require(set(fc.strategy) == set(STRATEGIES) and not fc.duplicated(
        ['target', 'UBIGEO', 'forecast_month', 'strategy']).any(), 'C9: estrategias o claves duplicadas.')
    for s, g in fc.groupby('strategy'):
        expected = {(t, u, m) for t in TARGETS for u in districts for m in EXTERNAL_MONTHS
                    if s != 'global_42' or u != excluded}
        require(set(zip(g.target, g.UBIGEO, g.forecast_month)) == expected, f'C9: claves incorrectas en {s}.')
    require(np.array_equal(fc.y_true.to_numpy(), look(fc, 0)), 'C9: observados distintos del panel.')
    for s, k in [('naive1', 1), ('naive_seasonal', 12)]:
        require(np.array_equal(fc[fc.strategy == s].y_pred.to_numpy(), look(fc[fc.strategy == s], k)),
                f'C9: {s} no coincide con el panel.')
    require(np.isfinite(fc.y_pred).all() and fc.y_pred.ge(0).all(), 'C9: pronósticos no válidos.')


# ----------------------------------------------------------------------------- estadística
def add_errors(fc, Q):
    fc = fc.copy()
    fc['absolute_error'] = (fc.y_true - fc.y_pred).abs()
    fc['squared_error'] = (fc.y_true - fc.y_pred) ** 2
    fc['scaled_error'] = fc.absolute_error / [Q[(t, u)] for t, u in zip(fc.target, fc.UBIGEO)]
    fc['year'] = fc.forecast_month.dt.year.astype(str)
    fc['regime'] = np.where(fc.forecast_month < BREAK_DATE, 'pre_break', 'post_break')
    return fc


def district_metrics(data, extra=()):
    keys = ['target', 'UBIGEO', 'strategy'] + list(extra)
    out = data.groupby(keys, observed=True, as_index=False).agg(
        n_months=('scaled_error', 'size'), MASE=('scaled_error', 'mean'),
        MAE=('absolute_error', 'mean'), MSE=('squared_error', 'mean'))
    out['RMSE'] = np.sqrt(out.pop('MSE'))
    return out


def snap_ties(delta):
    """Diferencias de valor absoluto menor que TIE_TOL se tratan como 0; no es equivalencia práctica."""
    if isinstance(delta, pd.Series):
        return delta.where(delta.abs() >= TIE_TOL, 0.0)
    return np.where(np.abs(delta) < TIE_TOL, 0.0, delta)


def diff(metric, a, b, target):
    """Por distrito: métrica_a − métrica_b (negativo: a tiene menor error)."""
    return snap_ties((metric.loc[target, a] - metric.loc[target, b]).sort_index())


def paired_summary(delta, tests=False):
    """Mediana de las diferencias, IC95 bootstrap de distritos, conteos y (opcional) Wilcoxon bilateral."""
    from scipy.stats import wilcoxon
    delta = snap_ties(np.asarray(delta, dtype=float))
    boot = np.median(np.random.default_rng(SEED).choice(delta, size=(N_BOOT, len(delta)), replace=True), axis=1)
    out = {'n_districts': len(delta), 'median_delta': float(np.median(delta)),
           'CI95_low': float(np.quantile(boot, .025)), 'CI95_high': float(np.quantile(boot, .975))}
    if tests:
        out['p_raw'] = 1.0 if np.all(delta == 0) else float(wilcoxon(
            delta, alternative='two-sided', zero_method='wilcox', method='auto').pvalue)
    out.update({'n_negative': int((delta < 0).sum()), 'n_positive': int((delta > 0).sum()),
                'n_ties': int((delta == 0).sum())})
    return out


def holm_two_targets(table):
    """Holm sobre Robo y Hurto dentro de cada contraste (no sobre todos los análisis)."""
    result = table.copy()
    for _, idx in result.groupby('contrast').groups.items():
        g = result.loc[idx].sort_values('p_raw')
        result.loc[g.index, 'p_holm'] = np.maximum.accumulate(np.minimum(1.0, g.p_raw.to_numpy() * [2, 1]))
    return result


# ----------------------------------------------------------------------------- tablas del paper
NAMES = {'naive1': 'Naïve-1', 'naive_seasonal': 'Seasonal naïve', 'ses': 'SES',
         'ses_pendiente': 'SES + ½ OLS slope (ad hoc)†', 'local_HL': 'Local XGB, levels',
         'global_HL': 'Global XGB, levels (H_L)', 'global_HG': 'Global XGB, levels (H_G)',
         'local_dz_HL': 'Local XGB, Δz', 'global_dz_HL': 'Global XGB, Δz (H_L5)',
         'global_dz_HG': 'Global XGB, Δz (H_G5)', 'ridge_dz': 'Ridge (global), Δz'}
STATUS = {'naive1': 'Reference', 'naive_seasonal': 'Reference', 'ses': 'Post hoc', 'ses_pendiente': 'Post hoc',
          'local_HL': 'Primary', 'global_HL': 'Primary', 'global_HG': 'Sensitivity', 'local_dz_HL': 'Exploratory',
          'global_dz_HL': 'Exploratory', 'global_dz_HG': 'Exploratory', 'ridge_dz': 'Exploratory'}
TARGET_EN = {'ROBO': 'Robbery', 'HURTO': 'Theft'}


def f3(x):
    """Cero exacto → '0'; si un valor distinto de cero redondea a 0.000, 4 decimales."""
    if x == 0:
        return '0'
    return f'{x:+.4f}' if round(x, 3) == 0 else f'{x:+.3f}'


def fmt(m, lo, hi):
    return f'{f3(m)} [{f3(lo)}, {f3(hi)}]'


def save_paper_table(df, folder, name, notes):
    """CSV + Markdown con notas al pie (generadas aquí para que no se desactualicen)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f'{name}.csv', index=False)
    head = ['| ' + ' | '.join(df.columns) + ' |', '|' + '|'.join(['---'] * len(df.columns)) + '|']
    rows = ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    text = '\n'.join(head + rows) + '\n\n' + '\n\n'.join(f'*{n}*' for n in notes) + '\n'
    (folder / f'{name}.md').write_text(text, encoding='utf-8')
    return text


def strategies_table(e8, strategies):
    rows = []
    for t in TARGETS:
        for s in strategies:
            r = e8[(e8.target == t) & (e8.strategy == s)].iloc[0]
            vs_n = '—' if s == 'naive1' else (f'{fmt(r.median_diff_vs_naive1, r.CI95_low_vs_naive1, r.CI95_high_vs_naive1)}'
                                             f' ({int(r.n_A_better_vs_naive1)})')
            vs_s = '—' if s == 'ses' else (f'{fmt(r.median_diff_vs_ses, r.CI95_low_vs_ses, r.CI95_high_vs_ses)}'
                                          f' ({int(r.n_A_better_vs_ses)})')
            rows.append({'Target': TARGET_EN[t], 'Strategy (A)': NAMES[s], 'Status': STATUS[s],
                         'Median MASE': f'{r.MASE_median:.3f}', 'Median MAE': f'{r.MAE_median:.1f}',
                         'A − Naïve-1': vs_n, 'A − SES': vs_s})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- reentrenar niveles (opcional)
def retrain_level_calibration(features, districts, grid):
    """Solo si REENTRENAR_NIVELES = True: rehace las 16,512 predicciones internas en niveles (≈ 8,448 ajustes)."""
    rows = []
    for t in TARGETS:
        d = features[features.target == t]
        for origin in CALIBRATION_MONTHS:
            train, test = d[d.fecha < origin], d[d.fecha == origin].sort_values('UBIGEO')
            for params in grid:
                for strategy in ['local', 'global']:
                    groups = [(train, test)] if strategy == 'global' else \
                        [(train[train.UBIGEO == u], test[test.UBIGEO == u]) for u in districts]
                    for tr, te in groups:
                        model = xgb(LEVEL_PARAMS, params['max_depth'], params['n_estimators'])
                        y_hat = to_counts_from_z(model.fit(tr[LEVEL_FEATURES], tr.z.astype(float)).predict(te[LEVEL_FEATURES]))
                        rows += [(t, str(u), origin, strategy, params['max_depth'], params['n_estimators'], yt, yp)
                                 for u, yt, yp in zip(te.UBIGEO, te.y.astype(float), y_hat)]
    return pd.DataFrame(rows, columns=['target', 'UBIGEO', 'origin', 'strategy', 'max_depth', 'n_estimators',
                                       'y_true', 'y_pred'])


def level_forecast(train, test, h):
    model = xgb(LEVEL_PARAMS, h['max_depth'], h['n_estimators']).fit(train[LEVEL_FEATURES], train.z.astype(float))
    return to_counts_from_z(model.predict(test[LEVEL_FEATURES]))


# ----------------------------------------------------------------------------- análisis del anexo
def e3_santa_maria(fc_all, Q, excluded):
    """E3: Global H_G con 42 vs 43 distritos, evaluados en los mismos 42."""
    sub = fc_all[((fc_all.strategy == 'global_HG') & (fc_all.UBIGEO != excluded)) | (fc_all.strategy == 'global_42')]
    m = district_metrics(sub).set_index(['target', 'strategy', 'UBIGEO']).MASE.sort_index()
    return holm_two_targets(pd.DataFrame([{'contrast': 'global42_minus_global43_HG', 'target': t,
                                           **paired_summary(diff(m, 'global_42', 'global_HG', t), tests=True)}
                                          for t in TARGETS]))


def e4_pre_post(main):
    """E4: D = (Global − Local)_post − (Global − Local)_pre por distrito; y la curva mensual post-quiebre."""
    pm = district_metrics(main[main.strategy.isin(['local_HL', 'global_HL'])], extra=['regime'])
    tab = pm.pivot(index=['target', 'UBIGEO'], columns=['strategy', 'regime'], values='MASE')
    e = pd.DataFrame(index=tab.index)
    for s, short in [('local_HL', 'Local'), ('global_HL', 'Global')]:
        for per, lab in [('pre_break', 'pre'), ('post_break', 'post')]:
            e[f'{short}_{lab}'] = tab[(s, per)]
    e['Delta_pre'], e['Delta_post'] = e.Global_pre - e.Local_pre, e.Global_post - e.Local_post
    e['D_post_minus_pre'] = e.Delta_post - e.Delta_pre
    rows = []
    for t, g in e.reset_index().groupby('target'):
        row = {'target': t, 'contrast': 'paired_change_post_minus_pre'}
        row.update({f'{n}_median': g[n].median() for n in ['Local_pre', 'Global_pre', 'Local_post', 'Global_post']})
        row.update({'median_delta_pre': g.Delta_pre.median(), 'median_delta_post': g.Delta_post.median()})
        rows.append({**row, **paired_summary(g.D_post_minus_pre, tests=True)})
    e4 = holm_two_targets(pd.DataFrame(rows)).rename(columns={'median_delta': 'median_D'})
    post = main[(main.forecast_month >= BREAK_DATE) & main.strategy.isin(['local_HL', 'global_HL'])].copy()
    post['k'] = (post.forecast_month.dt.year - BREAK_DATE.year) * 12 + post.forecast_month.dt.month - BREAK_DATE.month
    pk = post.pivot(index=['target', 'UBIGEO', 'k'], columns='strategy', values='scaled_error')
    pk['delta'] = pk.global_HL - pk.local_HL
    curve = pk.reset_index().groupby(['target', 'k'], as_index=False).agg(
        median_delta=('delta', 'median'), q1=('delta', lambda x: x.quantile(.25)),
        q3=('delta', lambda x: x.quantile(.75)), n_districts=('UBIGEO', 'nunique'))
    return e4, curve


def e7_by_period(main):
    """E7: MASE relativo a naïve-1 y proporción de distritos mejores que naïve-1, por año y por régimen."""
    out = []
    for col in ['year', 'regime']:
        m = district_metrics(main, extra=[col])
        base = m[m.strategy == 'naive1'][['target', 'UBIGEO', col, 'MASE']].rename(columns={'MASE': 'MASE_naive1'})
        m = m.merge(base, on=['target', 'UBIGEO', col])
        m['rel_MASE'] = np.where(m.MASE_naive1 > 0, m.MASE / m.MASE_naive1.where(m.MASE_naive1 > 0), np.nan)
        m['better'] = snap_ties(m.MASE - m.MASE_naive1) < 0
        g = m.groupby(['target', 'strategy', col]).agg(
            MASE_median=('MASE', 'median'), rel_MASE_median=('rel_MASE', 'median'), n_ratio_valid=('rel_MASE', 'count'),
            n_naive1_perfect=('MASE_naive1', lambda s: int((s == 0).sum())),
            share_better_than_naive1=('better', 'mean'), n_districts=('UBIGEO', 'nunique')).reset_index()
        out.append(g.rename(columns={col: 'period'}).assign(period_type=col))
    return pd.concat(out, ignore_index=True)


PAIRS = {'niveles': ('global_HL', 'local_HL'), 'dz': ('global_dz_HL', 'local_dz_HL')}


def e9_volume(panel, mase):
    """E9: correlación de Spearman entre log(volumen 2018–22) y diff(Global, Local), con IC bootstrap."""
    from scipy.stats import spearmanr
    volume = panel[panel.fecha < '2023-01-01'].groupby(['target', 'UBIGEO']).y.median()
    rows, pts = [], []
    for t in TARGETS:
        for form, (g, l) in PAIRS.items():
            d = diff(mase, g, l, t)
            v = volume.loc[t].reindex(d.index)
            idx = np.random.default_rng(SEED).integers(0, len(d), size=(N_BOOT, len(d)))
            boot = [spearmanr(np.log1p(v.values[k]), d.values[k]).statistic for k in idx]
            terc = pd.qcut(v, 3, labels=['bajo', 'medio', 'alto'])
            row = {'target': t, 'formulation': form, 'spearman_rho': float(spearmanr(np.log1p(v), d).statistic),
                   'rho_CI95_low': float(np.nanquantile(boot, .025)), 'rho_CI95_high': float(np.nanquantile(boot, .975))}
            for lab in ['bajo', 'medio', 'alto']:
                row[f'median_diff_tercil_{lab}'] = float(d[terc == lab].median())
                row[f'n_global_better_tercil_{lab}'] = int((d[terc == lab] < 0).sum())
                row[f'n_tercil_{lab}'] = int((terc == lab).sum())
            rows.append(row)
            pts.append(pd.DataFrame({'target': t, 'formulation': form, 'UBIGEO': d.index, 'volume': v.values,
                                     'diff_global_minus_local': d.values}))
    return pd.DataFrame(rows), pd.concat(pts, ignore_index=True)


def e10_extras(main, scores):
    """E10 (anexo): cuánto se aparta cada estrategia de naïve-1, y el top-5 de calibración."""
    nv = main[main.strategy == 'naive1'][['target', 'UBIGEO', 'forecast_month', 'y_pred']].rename(columns={'y_pred': 'nv'})
    dev = main[main.strategy != 'naive1'].merge(nv, on=['target', 'UBIGEO', 'forecast_month'])
    dev['correction'] = (dev.y_pred - dev.nv).abs() / dev.nv.clip(lower=1)
    corr = dev.groupby(['target', 'strategy']).correction.median().rename('median_rel_correction_vs_naive1').reset_index()
    cal = scores[scores.strategy.isin(['local', 'global'])].copy()
    cal['rank'] = cal.groupby(['formulation', 'target', 'strategy']).Score.rank(method='first')
    top5 = (cal[cal['rank'] <= 5][['formulation', 'target', 'strategy', 'rank', 'max_depth', 'n_estimators', 'Score']]
            .sort_values(['formulation', 'target', 'strategy', 'rank']))
    return corr, top5


# ============================================================================= ETAPAS (lo que llaman las notebooks)
ES = {'ROBO': 'Robo', 'HURTO': 'Hurto'}
LABEL_ES = {'naive1': 'Naïve-1', 'naive_seasonal': 'Naïve estacional', 'ses': 'SES',
            'ses_pendiente': 'SES + ½ pendiente (ad hoc)', 'local_HL': 'Local · niveles',
            'global_HL': 'Global · niveles (H_L)', 'global_HG': 'Global · niveles (H_G)', 'global_42': 'Global · 42 distritos',
            'local_dz_HL': 'Local · Δz', 'global_dz_HL': 'Global · Δz (H_L5)', 'global_dz_HG': 'Global · Δz (H_G5)',
            'ridge_dz': 'Ridge · Δz'}


def num(x, d=3):
    return '0' if x == 0 else f'{x:+.{d}f}'.replace('-', '−')


def ic(m, lo, hi, d=3):
    """'−0.081 [−0.099; −0.058]' con signo tipográfico."""
    return f'{num(m, d)} [{num(lo, d)}; {num(hi, d)}]'


def setup(project_dir, results_dir="resultados"):
    """Rutas, aviso de versiones (C1) y datos auditados (C2)."""
    p = project(project_dir, results_dir)
    env = check_env()
    panel, Q = load_inputs(p)
    districts = sorted(panel.UBIGEO.astype(str).unique())
    return p, env, panel, Q, districts


def panel_summary(panel):
    g = panel.groupby('target')
    return pd.DataFrame({'Distritos': g.UBIGEO.nunique(), 'Meses': g.fecha.nunique(),
                         'Desde': g.fecha.min().dt.strftime('%Y-%m'), 'Hasta': g.fecha.max().dt.strftime('%Y-%m'),
                         'Denuncias/mes (mediana Lima)': panel.groupby(['target', 'fecha']).y.sum().groupby('target').median()
                         .round(0).astype(int)}).rename(index=ES).loc[['Robo', 'Hurto']]


# ----------------------------------------------------------------------------- Notebook 02: calibración 2022
LEVEL_GRID = [{'max_depth': d, 'n_estimators': n} for d in [2, 3, 4, 6] for n in [100, 300]]
DZ_GRID = {'max_depth': [1, 2, 4, 6], 'n_estimators': [25, 50, 100, 300], 'ridge_alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}


def calibrate_levels(p, features, districts, Q, retrain=False):
    """Parte A: score 2022 de las 8 configuraciones en niveles (por defecto, predicciones de la corrida v2; C3)."""
    if retrain:
        pred = retrain_level_calibration(features, districts, LEVEL_GRID)
    else:
        pred = load_evidence(p, 'calibracion_niveles_original/notebook02_internal_predictions.csv')
        pred['origin'] = pd.to_datetime(pred.origin)
    pred = pred[['target', 'UBIGEO', 'origin', 'strategy', 'max_depth', 'n_estimators', 'y_true', 'y_pred']]
    _, scores = calibration_scores(pred, ['target', 'strategy', 'max_depth', 'n_estimators'], Q)
    selected = (scores.sort_values(['target', 'strategy', 'Score', 'max_depth', 'n_estimators'])
                .groupby(['target', 'strategy']).head(1))
    return scores, selected


def calibrate_dz(features, districts, Q):
    """Parte B: rolling-origin en 2022 para XGBoost Δz (Local y Global) y ridge."""
    staged_check(features)                                              # C4
    depths, trees, alphas = DZ_GRID['max_depth'], DZ_GRID['n_estimators'], DZ_GRID['ridge_alpha']
    rows = []
    for target in TARGETS:
        d = features[features.target == target]
        for origin in CALIBRATION_MONTHS:
            print(f'Calibración {target}: {origin:%Y-%m}', flush=True)
            train, test = d[d.fecha < origin], d[d.fecha == origin].sort_values('UBIGEO')
            arms = [('global', train, test)] + [('local', train[train.UBIGEO == u], test[test.UBIGEO == u])
                                                 for u in districts]
            for depth in depths:
                for strategy, tr, te in arms:
                    model = xgb(DZ_PARAMS, depth, max(trees)).fit(tr[DZ_FEATURES], tr.dz)
                    for n in trees:   # un modelo de 300 árboles, leído con sus primeros n (C4)
                        y_hat = to_counts_from_dz(te.lag_1, model.predict(te[DZ_FEATURES], iteration_range=(0, n)))
                        rows += [(target, origin, str(u), strategy, f'd{depth}_n{n}', yt, float(yp))
                                 for u, yt, yp in zip(te.UBIGEO, te.y.astype(float), y_hat)]
            for alpha in alphas:
                y_hat = ridge_forecast(train, test, alpha)
                rows += [(target, origin, str(u), 'ridge', f'a{alpha:g}', yt, float(yp))
                         for u, yt, yp in zip(test.UBIGEO, test.y.astype(float), y_hat)]
    pred = pd.DataFrame(rows, columns=['target', 'origin', 'UBIGEO', 'strategy', 'config_id', 'y_true', 'y_pred'])
    return score_dz_predictions(pred, Q)


def score_dz_predictions(pred, Q):
    """Reconstruye scores y selección Δz/Ridge desde predicciones internas, sin entrenar."""
    depths, trees, alphas = DZ_GRID['max_depth'], DZ_GRID['n_estimators'], DZ_GRID['ridge_alpha']
    pred, scores = calibration_scores(pred, ['target', 'strategy', 'config_id'], Q)
    grid = pd.DataFrame([{'config_id': f'd{d}_n{n}', 'max_depth': d, 'n_estimators': n, 'alpha': np.nan}
                         for d in depths for n in trees] +
                        [{'config_id': f'a{a:g}', 'max_depth': np.nan, 'n_estimators': np.nan, 'alpha': a} for a in alphas])
    scores = scores.merge(grid, on='config_id').assign(alpha_desc=lambda x: -x.alpha)
    selected = (scores.sort_values(['target', 'strategy', 'Score', 'max_depth', 'n_estimators', 'alpha_desc'])
                .groupby(['target', 'strategy']).head(1).drop(columns='alpha_desc'))
    return pred, scores.drop(columns='alpha_desc'), selected


def freeze_config(p, env, panel, level_scores, level_selected, dz_pred, dz_scores, dz_selected, retrain=False):
    """Congela la configuración y guarda scores, predicciones de 2022 y manifiesto."""
    ref = dz_pred[(dz_pred.strategy == 'ridge') & (dz_pred.config_id == 'a1')].copy()
    y = panel.assign(UBIGEO=panel.UBIGEO.astype(str)).set_index(['target', 'UBIGEO', 'fecha']).y.astype(float)
    ref['naive1'] = y.reindex(pd.MultiIndex.from_arrays([ref.target, ref.UBIGEO, ref.origin - pd.DateOffset(months=1)])).values
    naive_2022 = ((ref.y_true - ref.naive1).abs() / ref.Q_j).groupby([ref.target, ref.UBIGEO]).mean().groupby(level=0).median()
    pick = lambda tab, t, s: tab[(tab.target == t) & (tab.strategy == s)].iloc[0]
    hp = lambda r: {'max_depth': int(r.max_depth), 'n_estimators': int(r.n_estimators), 'Score': float(r.Score)}
    selected = {t: {'H_L': hp(pick(level_selected, t, 'local')), 'H_G': hp(pick(level_selected, t, 'global')),
                    'H_L5': hp(pick(dz_selected, t, 'local')), 'H_G5': hp(pick(dz_selected, t, 'global')),
                    'ridge_alpha': float(pick(dz_selected, t, 'ridge').alpha),
                    'ridge_Score': float(pick(dz_selected, t, 'ridge').Score),
                    'naive1_Score_2022': float(naive_2022[t])} for t in TARGETS}
    config = {'version': 'v0.8.5', 'input_hashes': INPUT_HASHES, 'selected': selected,
              'level_params': LEVEL_PARAMS, 'dz_params': DZ_PARAMS, 'level_features': LEVEL_FEATURES,
              'dz_features': DZ_FEATURES, 'ridge_features': RIDGE_FEATURES, 'level_grid': LEVEL_GRID, 'dz_grid': DZ_GRID,
              'calibration_months': [str(m.date()) for m in CALIBRATION_MONTHS],
              'score_definition': 'mediana de 43 distritos del MASE del distrito (media de 12 meses de 2022)',
              'tie_break': 'menor max_depth, luego menor n_estimators; ridge: mayor alpha',
              'level_calibration_origin': 'reentrenado' if retrain else 'reutilizado_de_la_corrida_v2',
              'e3_excluded_ubigeo': '150138'}
    scores = pd.concat([level_scores.assign(formulation='niveles', config_id=lambda x: 'd' + x.max_depth.astype(str)
                                            + '_n' + x.n_estimators.astype(str)), dz_scores.assign(formulation='dz')],
                       ignore_index=True)
    write_json(p.config, config)
    write_csv(p.scores, scores)
    write_csv(p.cal / 'calibracion_predicciones_dz_2022.csv', dz_pred)
    save_manifest(p.cal, env, [p.config, p.scores, p.cal / 'calibracion_predicciones_dz_2022.csv'])
    return selected


def selection_table(selected):
    """Qué se eligió para cada estrategia y su score de 2022 (menor = menos error)."""
    cell = lambda h: f"{h['max_depth']} × {h['n_estimators']} · {h['Score']:.3f}"
    rows = {name: {'Estatus': st, **{ES[t]: cell(selected[t][key]) for t in TARGETS}}
            for key, name, st in [('H_L', 'Local · niveles (H_L)', 'Principal'), ('H_G', 'Global · niveles (H_G)', 'Sensibilidad'),
                                  ('H_L5', 'Local · Δz (H_L5)', 'Exploratorio'), ('H_G5', 'Global · Δz (H_G5)', 'Exploratorio')]}
    rows['Ridge · Δz (α)'] = {'Estatus': 'Exploratorio',
                              **{ES[t]: f"α = {selected[t]['ridge_alpha']:g} · {selected[t]['ridge_Score']:.3f}" for t in TARGETS}}
    rows['Naïve-1 (solo referencia)'] = {'Estatus': 'Referencia',
                                         **{ES[t]: f"— · {selected[t]['naive1_Score_2022']:.3f}" for t in TARGETS}}
    return pd.DataFrame(rows).T.rename_axis('profundidad × árboles · score 2022')


# ----------------------------------------------------------------------------- Notebook 03: pronósticos 2023–2026
COLS = ['target', 'UBIGEO', 'DIST_HECHO', 'cutoff', 'forecast_month', 'strategy', 'y_true', 'y_pred']


def _rows(test, strategy, values, month):
    out = test[['target', 'UBIGEO', 'DIST_HECHO']].assign(UBIGEO=lambda x: x.UBIGEO.astype(str))
    out['cutoff'], out['forecast_month'], out['strategy'] = month - pd.DateOffset(months=1), month, strategy
    out['y_true'], out['y_pred'] = test.y.to_numpy(float), np.asarray(values, dtype='float64')
    return out[COLS]


def forecast_all(p, panel, districts, config, retrain_levels=False):
    """Genera Δz y referencias con historia anterior a cada mes; por defecto reutiliza los niveles v2."""
    check_config_contract(config)
    S, smm = config['selected'], config['e3_excluded_ubigeo']
    featured = build_features(panel, districts)
    Y = {(r.target, str(r.UBIGEO), r.fecha): float(r.y) for r in panel.itertuples()}
    pieces = []
    for target in TARGETS:
        s, d = S[target], featured[featured.target == target]
        for month in EXTERNAL_MONTHS:
            print(f'Pronósticos {target}: {month:%Y-%m}', flush=True)
            train, test = d[d.fecha < month], d[d.fecha == month].sort_values('UBIGEO').reset_index(drop=True)
            require(train.fecha.max() == month - pd.DateOffset(months=1), 'C8: corte temporal incorrecto.')
            by_district = [(train[train.UBIGEO == u], test[test.UBIGEO == u]) for u in districts]
            f = {'local_dz_HL': [float(dz_forecast(DZ_PARAMS, tr, te, s['H_L5'])[0]) for tr, te in by_district],
                 'global_dz_HL': dz_forecast(DZ_PARAMS, train, test, s['H_L5']),
                 'global_dz_HG': dz_forecast(DZ_PARAMS, train, test, s['H_G5']),
                 'ridge_dz': ridge_forecast(train, test, s['ridge_alpha']),
                 'naive1': [Y[(target, str(u), month - pd.DateOffset(months=1))] for u in test.UBIGEO],
                 'naive_seasonal': [Y[(target, str(u), month - pd.DateOffset(months=12))] for u in test.UBIGEO]}
            if retrain_levels:
                f['local_HL'] = [float(level_forecast(tr, te, s['H_L'])[0]) for tr, te in by_district]
                f['global_HL'], f['global_HG'] = level_forecast(train, test, s['H_L']), level_forecast(train, test, s['H_G'])
                keep = test.UBIGEO.astype(str) != smm
                pieces.append(_rows(test[keep], 'global_42', level_forecast(
                    train[train.UBIGEO.astype(str) != smm], test[keep], s['H_G']), month))
            pieces += [_rows(test, k, v, month) for k, v in f.items()]
    stat = []                                  # SES y SES + ½ pendiente, por distrito, sobre log1p(y)
    for (target, u), g in panel.groupby(['target', 'UBIGEO']):
        dates, y = g.fecha.to_numpy(), g.y.astype(float).to_numpy()
        z = np.log1p(y)
        for month in EXTERNAL_MONTHS:
            i = int(np.flatnonzero(dates == np.datetime64(month))[0])        # z[:i]: solo meses anteriores
            level = ses_level(z[:i])
            for name, value in [('ses', level), ('ses_pendiente', ses_slope_level(z[:i], level))]:
                stat.append([target, str(u), g.DIST_HECHO.iloc[0], month - pd.DateOffset(months=1), month, name,
                             y[i], max(0.0, float(np.expm1(value)))])
    pieces.append(pd.DataFrame(stat, columns=COLS))
    if not retrain_levels:
        pieces.append(level_evidence_forecasts(p, S)[COLS])                   # C6, C7
    fc = pd.concat(pieces, ignore_index=True)
    fc['level_source'] = np.where(fc.strategy.isin(LEVEL_H), 'reentrenado' if retrain_levels else 'evidencia_v2',
                                  'calculado_aqui')
    fc = fc.sort_values(['strategy', 'target', 'forecast_month', 'UBIGEO']).reset_index(drop=True)
    check_forecasts(fc, panel, districts, smm)                                # C9
    return fc


def save_forecasts(p, env, fc, retrain_levels=False):
    write_csv(p.forecasts, fc)
    save_manifest(p.forecast, env, [p.forecasts], config_sha256=sha256_file(p.config), reentrenar_niveles=retrain_levels)


def forecast_summary(fc):
    """Una fila por estrategia: cuántos pronósticos, de dónde vienen y su estatus."""
    status = {'local_HL': 'Principal', 'global_HL': 'Principal', 'global_HG': 'Sensibilidad', 'global_42': 'Sensibilidad',
              'naive1': 'Referencia', 'naive_seasonal': 'Referencia', 'ses': 'Post hoc', 'ses_pendiente': 'Post hoc',
              'local_dz_HL': 'Exploratorio', 'global_dz_HL': 'Exploratorio', 'global_dz_HG': 'Exploratorio',
              'ridge_dz': 'Exploratorio'}
    origin = {'evidencia_v2': 'corrida v2 (verificada)', 'calculado_aqui': 'calculado en la corrida registrada', 'reentrenado': 'reentrenado en la corrida registrada'}
    g = fc.groupby('strategy')
    out = pd.DataFrame({'Estatus': pd.Series(status), 'Pronósticos': g.size(),
                        'Origen': g.level_source.first().map(origin)}).loc[STRATEGIES]
    return out.rename(index=LABEL_ES)


# ----------------------------------------------------------------------------- Notebook 04: resultados
def load_results_inputs(p, Q):
    """C5 + C10: lee entradas verificadas y añade errores. No borra archivos."""
    config = load_config(p)
    fc_all = add_errors(load_forecasts(p), Q)
    return config, fc_all, fc_all[fc_all.strategy != 'global_42']


CONTRASTS = [('E1', 'global_HL', 'local_HL'), ('E1b', 'global_HG', 'local_HL'), ('E5', 'global_dz_HL', 'local_dz_HL'),
             ('E5b', 'global_dz_HG', 'local_dz_HL'), ('E5c_local', 'local_dz_HL', 'local_HL'),
             ('E5c_global', 'global_dz_HG', 'global_HG')]
ALL = ['naive1', 'naive_seasonal', 'ses', 'ses_pendiente', 'local_HL', 'global_HL', 'global_HG',
       'local_dz_HL', 'global_dz_HL', 'global_dz_HG', 'ridge_dz']


def global_vs_local(MASE):
    """E1, E1b, E5, E5b, E5c: diferencia pareada por distrito, IC95 bootstrap, Wilcoxon y Holm por contraste."""
    rows, pairs = [], []
    for exp, a, b in CONTRASTS:
        for t in TARGETS:
            d = diff(MASE, a, b, t)
            pairs.append(pd.DataFrame({'experiment': exp, 'target': t, 'UBIGEO': d.index, 'MASE_A': MASE.loc[t, a].values,
                                       'MASE_B': MASE.loc[t, b].values, 'delta': d.values}))
            rows.append({'experiment': exp, 'contrast': f'{a}_minus_{b}', 'target': t, **paired_summary(d, tests=True)})
    return holm_two_targets(pd.DataFrame(rows)), pd.concat(pairs, ignore_index=True)


def vs_references(MASE, MAE):
    """E2/E8: cada estrategia frente a naïve-1 y frente a SES; E6: frente a naïve estacional y ridge (sin IC)."""
    rows = []
    for t in TARGETS:
        for a in ALL:
            row = {'target': t, 'strategy': a, 'MASE_median': MASE.loc[t, a].median(), 'MAE_median': MAE.loc[t, a].median()}
            for b in [x for x in ['naive1', 'ses'] if x != a]:
                s = paired_summary(diff(MASE, a, b, t))
                row.update({f'median_diff_vs_{b}': s['median_delta'], f'CI95_low_vs_{b}': s['CI95_low'],
                            f'CI95_high_vs_{b}': s['CI95_high'], f'n_A_better_vs_{b}': s['n_negative'],
                            f'n_ties_vs_{b}': s['n_ties']})
            rows.append(row)
    e6 = pd.DataFrame([{'target': t, 'model': m, 'baseline': b, 'median_delta_model_minus_baseline': d.median(),
                        'n_model_better': int((d < 0).sum()), 'n_model_worse': int((d > 0).sum())}
                       for t in TARGETS for m, b in [(m, 'naive_seasonal') for m in ALL[4:]] + [(m, 'ridge_dz') for m in ALL[7:10]]
                       for d in [diff(MASE, m, b, t)]])
    return pd.DataFrame(rows), e6


def by_year(main):
    """E10: mediana por año de la diferencia pareada Global − Local (niveles y Δz)."""
    m = main.groupby(['target', 'strategy', 'year', 'UBIGEO']).scaled_error.mean()
    return pd.DataFrame([{'target': t, 'year': y, 'formulation': form, 'median_diff_global_minus_local': d.median(),
                          'n_global_better': int((d < 0).sum()), 'n_districts': len(d)}
                         for t in TARGETS for y in ['2023', '2024', '2025', '2026'] for form, (g, l) in PAIRS.items()
                         for d in [snap_ties(m.loc[t, g, y] - m.loc[t, l, y])]])


# Vistas cortas (lo que se muestra en la notebook)
def view_contrasts(contrasts, experiments):
    names = {'E1': 'E1 · Global − Local, niveles (misma H_L)', 'E1b': 'E1b · Global (H_G) − Local, niveles',
             'E5': 'E5 · Global − Local, Δz (misma H_L5)', 'E5b': 'E5b · Global (H_G5) − Local, Δz',
             'E5c_local': 'E5c · Local: Δz − niveles', 'E5c_global': 'E5c · Global (H_G): Δz − niveles'}
    c = contrasts[contrasts.experiment.isin(experiments)]
    return pd.DataFrame({'Contraste': c.experiment.map(names), 'Delito': c.target.map(ES),
                         'Mediana ΔMASE [IC95]': [ic(r.median_delta, r.CI95_low, r.CI95_high) for r in c.itertuples()],
                         'A mejor en': [f'{r.n_negative}/43' for r in c.itertuples()],
                         'p Holm': c.p_holm.map(lambda x: f'{x:.2g}')}).set_index(['Contraste', 'Delito'])


def view_references(e8, strategies):
    e = e8[e8.strategy.isin(strategies)]
    out = pd.DataFrame({'Estrategia (A)': e.strategy.map(LABEL_ES), 'Delito': e.target.map(ES),
                        'MASE mediano': e.MASE_median.round(3),
                        'A − Naïve-1 [IC95]': [ic(r.median_diff_vs_naive1, r.CI95_low_vs_naive1, r.CI95_high_vs_naive1)
                                               if r.strategy != 'naive1' else '—' for r in e.itertuples()],
                        'Mejor que Naïve-1': [f'{int(r.n_A_better_vs_naive1)}/43' if r.strategy != 'naive1' else '—'
                                              for r in e.itertuples()],
                        'A − SES [IC95]': [ic(r.median_diff_vs_ses, r.CI95_low_vs_ses, r.CI95_high_vs_ses)
                                           if r.strategy != 'ses' else '—' for r in e.itertuples()]})
    out['_o'] = out['Estrategia (A)'].map({LABEL_ES[s]: i for i, s in enumerate(strategies)})
    return out.sort_values(['Delito', '_o'], ascending=[False, True]).drop(columns='_o').set_index(['Delito', 'Estrategia (A)'])


def view_by_year(e10):
    v = e10.assign(cell=[f"{num(m)} ({n}/43)" for m, n in
                         zip(e10.median_diff_global_minus_local, e10.n_global_better)],
                   Delito=e10.target.map(ES), Formulación=e10.formulation.map({'niveles': 'Niveles', 'dz': 'Δz'}))
    return v.pivot(index=['Delito', 'Formulación'], columns='year', values='cell').loc[['Robo', 'Hurto']]


def sentence_e1(contrasts):
    r = contrasts[contrasts.experiment == 'E1'].set_index('target')
    return ('Global tuvo menor error que Local en niveles: ' + '; '.join(
        f"{ES[t]} {ic(r.loc[t].median_delta, r.loc[t].CI95_low, r.loc[t].CI95_high)}, "
        f"Global mejor en {r.loc[t].n_negative} de 43 distritos" for t in TARGETS) + '.')


def overview(dm):
    """MASE mediano de cada estrategia (descriptivo; para decidir quién gana se usan diferencias pareadas)."""
    v = dm.groupby(['strategy', 'target']).MASE.median().unstack('target')[TARGETS].rename(columns=ES)
    return v.loc[ALL].rename(index=LABEL_ES).round(3).rename_axis('MASE mediano (43 distritos)')


def annex(p, Q, panel, fc_all, main, MASE, config, tab):
    """Análisis del anexo (E3, E4, E7, E9, E10 extra); guarda sus tablas y devuelve lo que usan figuras y tablas."""
    e3 = e3_santa_maria(fc_all, Q, config['e3_excluded_ubigeo'])
    e4, e4_curve = e4_pre_post(main)
    e7 = e7_by_period(main)
    e9, e9_districts = e9_volume(panel, MASE)
    e10_corr, e10_top5 = e10_extras(main, pd.read_csv(p.scores))
    for name, df in [('E3_santa_maria', e3), ('E4_pre_post', e4), ('E4_curva_post_quiebre', e4_curve),
                     ('E7_por_periodo', e7), ('E9_volumen', e9), ('E10_correccion_vs_naive1', e10_corr),
                     ('E10_calibracion_top5', e10_top5)]:
        write_csv(tab / f'{name}.csv', df)
    return {'e3': e3, 'e4': e4, 'e9': e9, 'e9_districts': e9_districts}


def view_annex(contrasts, ax):
    e4, e3 = ax['e4'].set_index('target').loc[TARGETS].reset_index(), ax['e3']
    rows = [{'Análisis': 'E4 · cambio post − pre (Global − Local)', 'Delito': ES[r.target],
             'Mediana [IC95]': ic(r.median_D, r.CI95_low, r.CI95_high), 'p Holm': f'{r.p_holm:.2g}'} for r in e4.itertuples()]
    rows += [{'Análisis': 'E3 · Global sin − con Santa María del Mar', 'Delito': ES[r.target],
              'Mediana [IC95]': ic(r.median_delta, r.CI95_low, r.CI95_high), 'p Holm': f'{r.p_holm:.2g}'} for r in e3.itertuples()]
    c = contrasts[contrasts.experiment.str.startswith('E5c')]
    rows += [{'Análisis': {'E5c_local': 'E5c · Local: Δz − niveles', 'E5c_global': 'E5c · Global (H_G): Δz − niveles'}[r.experiment],
              'Delito': ES[r.target], 'Mediana [IC95]': ic(r.median_delta, r.CI95_low, r.CI95_high),
              'p Holm': f'{r.p_holm:.2g}'} for r in c.itertuples()]
    return pd.DataFrame(rows).set_index(['Análisis', 'Delito'])


def save_paper_tables(p, folder, contrasts, e8, e10, ax):
    """Tablas del paper (CSV + Markdown con notas al pie). Devuelve la lista de archivos."""
    cells = ('The A − B columns report the median of 43 paired district differences MASE(A) − MASE(B) [95% bootstrap CI over districts] '
             '(districts where A has lower MASE). Negative favours A. Absolute paired differences below 1e-12 are treated as numerical ties. A displayed "0" denotes zero after this tolerance, not practical equivalence.')
    ties = {t: int(e8[(e8.target == t) & (e8.strategy == 'ses')].n_ties_vs_naive1.iloc[0]) for t in TARGETS}
    save_paper_table(strategies_table(e8, ['naive1', 'ses', 'local_HL', 'global_HL', 'local_dz_HL', 'global_dz_HL',
                                           'global_dz_HG']), folder, 'table1_strategies', [
        cells, f'SES vs naïve-1 has {ties["ROBO"]} ties in Robbery and {ties["HURTO"]} in Theft after the numerical tolerance. SES with α = 1 reproduces naïve-1; α is not stored for each forecast.',
        'Median MASE is descriptive and can rank strategies differently from paired differences. The A − SES column is '
        'post hoc. H_L/H_G: hyperparameters tuned on 2022 (levels); H_L5/H_G5: same for Δz. Full table: Table A2.'])
    t2 = pd.DataFrame([{'Contrast (Global − Local)': lab, 'Status': stat, 'Target': TARGET_EN[t],
                        'Median ΔMASE [95% CI]': fmt(r.median_delta, r.CI95_low, r.CI95_high),
                        'Global better (of 43)': int(r.n_negative), 'Holm p': f'{r.p_holm:.2g}'}
                       for exp, lab, stat in [('E1', 'E1 · Levels, shared H_L', 'Primary'),
                                              ('E1b', 'E1b · Levels, Global H_G', 'Sensitivity'),
                                              ('E5', 'E5 · Δz, shared H_L5', 'Exploratory'),
                                              ('E5b', 'E5b · Δz, Global H_G5', 'Exploratory')]
                       for t in TARGETS for r in [contrasts[(contrasts.experiment == exp) & (contrasts.target == t)].iloc[0]]])
    save_paper_table(t2, folder, 'table2_global_vs_local', [
        'Median of 43 paired district differences [95% bootstrap CI]. Negative favours Global. Holm p: two-sided Wilcoxon, '
        'Holm across the two targets within each contrast; not adjusted across contrasts. E1 is the primary contrast; E5 and E5b remain exploratory.',
        'These results use the registered forecast file. No independent cross-machine reproduction is established here.'])
    e4 = ax['e4'].set_index('target').loc[TARGETS].reset_index()
    a1 = [{'Analysis': 'E4 · post − pre recording change', 'Status': 'Secondary, non-causal', 'Target': TARGET_EN[r.target],
           'Global − Local, pre / post': f'{f3(r.median_delta_pre)} / {f3(r.median_delta_post)}',
           'Median paired change [95% CI]': fmt(r.median_D, r.CI95_low, r.CI95_high), 'Holm p': f'{r.p_holm:.2g}'}
          for r in e4.itertuples()] + \
         [{'Analysis': 'E3 · Global without − with Santa María del Mar', 'Status': 'Sensitivity', 'Target': TARGET_EN[r.target],
           'Global − Local, pre / post': '—', 'Median paired change [95% CI]': fmt(r.median_delta, r.CI95_low, r.CI95_high),
           'Holm p': f'{r.p_holm:.2g}'} for r in ax['e3'].itertuples()]
    save_paper_table(pd.DataFrame(a1), folder, 'tableA1_e4_e3', [
        'E4: per district, (Global − Local)_post − (Global − Local)_pre, levels with shared H_L; pre 2023-01–2024-10, post '
        '2024-11–2026-07. Not a causal effect. Monthly post-change curve, relative to this table: ../../tablas/E4_curva_post_quiebre.csv.',
        'E3: Global H_G trained on 42 vs 43 districts, evaluated on the same 42. In Robbery the CI is positive while the Holm '
        'p is ≈ 0.33: they answer different questions (median vs signed ranks). A small change; a high p does not show '
        'equivalence.'])
    save_paper_table(strategies_table(e8, ALL), folder, 'tableA2_all_strategies', [
        cells, '† Ad hoc reference kept from the internal registry: SES level plus half the OLS slope of log1p(y); '
        'not the standard Theta method. Differences are reported in the A − SES column; no practical equivalence margin was defined.'])
    t3 = e10.assign(cell=[f'{f3(m)} ({n})' for m, n in zip(e10.median_diff_global_minus_local, e10.n_global_better)],
                    Formulation=e10.formulation.map({'niveles': 'Levels (H_L)', 'dz': 'Δz (H_L5)'}),
                    Target=e10.target.map(TARGET_EN)).pivot(index=['Target', 'Formulation'], columns='year', values='cell')
    save_paper_table(t3.reset_index().rename_axis(columns=None), folder, 'tableA3_annual', [
        'Post hoc descriptive (E10). Median annual paired difference MASE(Global) − MASE(Local) (districts where Global is '
        'lower, of 43). 2026: January–July.'])
    a4 = contrasts[contrasts.experiment.str.startswith('E5c')]
    save_paper_table(pd.DataFrame([{'Contrast': r.contrast, 'Target': TARGET_EN[r.target],
                                    'Median ΔMASE [95% CI]': fmt(r.median_delta, r.CI95_low, r.CI95_high),
                                    'A better (of 43)': r.n_negative, 'Holm p': f'{r.p_holm:.2g}'} for r in a4.itertuples()]),
                     folder, 'tableA4_dz_vs_levels', ['Exploratory (E5c): Δz specification minus level specification. Negative favours Δz. The response, predictors and calibration grid differ, so this is not an isolated effect of changing the response. Intervals use paired district bootstrap; p values use two-sided Wilcoxon with Holm across the two targets within each contrast.'])
    return sorted(x.name for x in Path(folder).glob('*.md'))


def view_selected(selected_df, labels):
    """Configuración ganadora por estrategia: 'profundidad × árboles (score 2022)'."""
    v = selected_df.assign(cell=[f'{int(r.max_depth)} × {int(r.n_estimators)}  (score {r.Score:.3f})'
                                 if not np.isnan(r.max_depth) else f'α = {r.alpha:g}  (score {r.Score:.3f})'
                                 for r in selected_df.itertuples()])
    v = v.pivot(index='strategy', columns='target', values='cell')[TARGETS].rename(columns=ES)
    return v.loc[list(labels)].rename(index=labels).rename_axis('Elegida con 2022')


# ----------------------------------------------------------------------------- revisión sin reentrenamiento

def require_training_environment(env):
    """El modo de entrenamiento debe utilizar las versiones acordadas; revisar no entrena."""
    wrong = {k: (v, env.get(k)) for k, v in EXPECTED_ENV.items() if env.get(k) != v}
    require(not wrong, f'No se inicia entrenamiento con versiones diferentes: {wrong}. '
            'Instale requirements_modelado.txt, reinicie el kernel o utilice MODO="revisar".')


def check_config_contract(config):
    """Las constantes utilizadas por el código deben coincidir con el JSON congelado."""
    expected = {'level_params': LEVEL_PARAMS, 'dz_params': DZ_PARAMS,
                'level_features': LEVEL_FEATURES, 'dz_features': DZ_FEATURES,
                'ridge_features': RIDGE_FEATURES, 'level_grid': LEVEL_GRID,
                'dz_grid': DZ_GRID,
                'calibration_months': [str(m.date()) for m in CALIBRATION_MONTHS],
                'input_hashes': INPUT_HASHES}
    for key, value in expected.items():
        require(config.get(key) == value, f'Configuración y código difieren en {key}. '
                'No continúe ni ajuste el archivo silenciosamente.')


def read_saved_calibration(p):
    """Verifica la calibración guardada y carga sus predicciones internas; no escribe inputs."""
    manifest = json.loads((p.cal / 'manifest.json').read_text())
    for name in ['configuracion_congelada.json', 'calibracion_scores.csv',
                 'calibracion_predicciones_dz_2022.csv']:
        require((p.cal / name).is_file(), f'Falta {p.cal / name}. Use el paquete completo.')
        require(sha256_file(p.cal / name) == manifest['output_hashes'][name],
                f'Cambió {name}. Restaure la calibración registrada, no el manifiesto.')
    config = load_config(p)
    check_config_contract(config)
    pred = pd.read_csv(p.cal / 'calibracion_predicciones_dz_2022.csv',
                       dtype={'UBIGEO': str}, parse_dates=['origin'], float_precision='round_trip')
    scores = pd.read_csv(p.scores, float_precision='round_trip')
    return config, pred, scores


def example_districts(panel, before='2023-01-01'):
    """Un distrito central por volumen previo, por target; nunca se selecciona por desempeño."""
    history = panel[panel.fecha < before]
    volume = history.groupby(['target', 'UBIGEO', 'DIST_HECHO'], as_index=False).y.median()
    chosen = []
    for target in TARGETS:
        part = volume[volume.target == target].sort_values(['y', 'UBIGEO'])
        row = part.iloc[len(part) // 2].to_dict()
        row['fecha_limite_volumen'] = before
        chosen.append(row)
    return pd.DataFrame(chosen).rename(columns={'y': 'mediana_volumen_previo'})
