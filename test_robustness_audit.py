from collections import Counter
from datetime import datetime, timedelta

from test_robustness_ui import _chronologie, _infrastructure, _mission, _app_test, _cliquer_calcul_robustesse
from robustness_logic import evaluer_robustesse


def test_audit_exhaustif_et_egalites():
    result = evaluer_robustesse(_chronologie(), _infrastructure(), [_mission()])
    for scenario in result['scenarios']:
        candidates = scenario['candidates_evaluated']
        assert len(candidates) == result['candidates_count']
        assert Counter(row['kind'] for row in candidates) == {'depart': 1, 'croisement': 1}
        maximum = max(row['score_min'] for row in candidates)
        tied = [row for row in candidates if abs(row['score_min'] - maximum) < 1e-6]
        assert scenario['tie_count'] == len(tied)
        assert len(scenario['points_ex_aequo']) == len(tied)
        assert scenario['sans_reserve']['retard_cumule_min'] == maximum


def test_ui_plafond_zero_invalide_et_labels():
    app = _app_test().run()
    app.checkbox(key='robustesse_reserve_active').check().run()
    app.multiselect(key='robustesse_reserve_gares').select('A').run()
    limit = app.number_input(key='robustesse_max_reserve_rames')
    assert limit.value == 1
    _cliquer_calcul_robustesse(app)
    assert not app.exception
    columns = {column for frame in app.dataframe for column in frame.value.columns}
    assert 'Retard cumulé — Sans réserve' in columns
    assert 'Retard cumulé — Avec réserve' in columns
    assert any('ex æquo' in caption.value for caption in app.caption)
    app.number_input(key='robustesse_max_reserve_rames').set_value(0).run()
    assert app.session_state['robustesse_resultats'] is None
    _cliquer_calcul_robustesse(app)
    assert not app.exception
    for frame in app.dataframe:
        assert 'str:T1#' not in frame.value.to_string()


def test_un_croisement_tardif_est_reellement_pire_que_tous_les_departs():
    base = datetime(2026, 1, 1)
    def journey(origin, destination, hour, minute, duration, wait=0):
        start = base.replace(hour=hour, minute=minute)
        arrival = start + timedelta(minutes=duration)
        departure = arrival + timedelta(minutes=wait)
        label = f'{origin} → {destination}'
        rows = [dict(start=start, end=arrival, origine=origin, terminus='X',
                     mission=label, is_mission_start=True)]
        if wait:
            rows.append(dict(start=arrival, end=departure, origine='X', terminus='X',
                             mission=label, is_mission_start=False))
        rows.append(dict(start=departure, end=departure + timedelta(minutes=duration),
                         origine='X', terminus=destination, mission=label, is_mission_start=False))
        return rows

    chronology = {
        'morning': journey('A', 'B', 8, 0, 10),
        '0': journey('A', 'B', 16, 26, 8, 5),
        '1': journey('A', 'B', 16, 42, 8, 2),
        '2': journey('B', 'A', 16, 25, 5, 5),
        '3': journey('B', 'A', 16, 20, 3),
    }
    scenario = evaluer_robustesse(chronology, _infrastructure(), [_mission()])['scenarios'][0]
    assert scenario['point']['kind'] == 'croisement'
    assert scenario['point']['time'] == base.replace(hour=16, minute=23)
    assert scenario['sans_reserve']['retard_cumule_min'] == 88
    assert scenario['tie_count'] == 1
    assert max(row['score_min'] for row in scenario['candidates_evaluated']
               if row['kind'] == 'depart') == 82
    assert max(row['score_min'] for row in scenario['candidates_evaluated']
               if row['time'].hour < 12) == 5
