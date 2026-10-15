import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from run import build_dataset_from_config
from vlmeval.dataset import Sen2Med
from vlmeval.dataset.sen2med import (
    DATASET_SHA256,
    DATASET_URL,
    SYSTEM_PROMPT,
    USER_PROMPT,
    _binary_metrics,
    clinician_majority_vote,
    majority_vote,
    parse_triage_response,
)
from vlmeval.inference import infer_data
from vlmeval.api.gpt import OpenAIWrapper
from vlmeval.smp import load


def _rater_columns(manage, visit, resource):
    decisions = {'manage': manage, 'visit': visit, 'resource': resource}
    values = {}
    for rater in range(1, 6):
        for task, decision in decisions.items():
            values[f'clin_rater{rater}_{task}'] = decision if rater <= 3 else ''
    return values


def _write_fixture(path):
    rows = [
        {
            'context_id': 'case-a',
            'perturbation': 'baseline',
            'dataset': 'askdocs',
            'clinical_context': 'Context A',
            'gold_manage': 'YES',
            'gold_visit': 'NO',
            'gold_resource': 'YES',
            **_rater_columns(1.0, 0.0, 1.0),
        },
        {
            'context_id': 'case-b',
            'perturbation': 'baseline',
            'dataset': 'oncqa',
            'clinical_context': 'Context B',
            'gold_manage': 'NO',
            'gold_visit': 'YES',
            'gold_resource': 'NO',
            **_rater_columns(0.0, 1.0, 0.0),
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_versioned_config_constructs_registered_dataset(tmp_path):
    data_path = tmp_path / 'dataset.csv'
    _write_fixture(data_path)
    config_path = Path(__file__).resolve().parents[1] / 'configs' / 'sen2med.json'
    config = load(str(config_path))['data']
    config['Sen2Med_baseline'].update({
        'data_file': str(data_path),
        'sample_limit': 1,
        'validate_release': False,
    })

    dataset = build_dataset_from_config(config, 'Sen2Med_baseline', strict=True)

    assert isinstance(dataset, Sen2Med)
    assert dataset.logical_sample_count == 2
    assert len(dataset) == 6
    assert list(dataset.data['repeat']) == [0, 1, 2, 0, 1, 2]
    assert list(dataset.data['index']) == [
        'case-a:baseline:seed0',
        'case-a:baseline:seed1',
        'case-a:baseline:seed2',
        'case-b:baseline:seed0',
        'case-b:baseline:seed1',
        'case-b:baseline:seed2',
    ]
    assert DATASET_URL.endswith('/46bee75f869a1fbe96966b66180f065d5bd12526/dataset.csv')
    assert DATASET_SHA256 == '381cc0296b525bf67ec52860caedf3c171bf94e41d1c993498e816b154e2e1fb'


def test_prompt_parser_and_seed_contract(tmp_path):
    data_path = tmp_path / 'dataset.csv'
    _write_fixture(data_path)
    dataset = Sen2Med(data_file=str(data_path), validate_release=False)

    prompt = dataset.build_prompt(0)
    assert prompt == [
        {'type': 'text', 'value': USER_PROMPT.format(clinical_context='Context A')},
    ]
    assert dataset.get_generation_kwargs(0) == {'seed': 0}
    assert dataset.get_generation_kwargs(2) == {'seed': 2}
    assert parse_triage_response(
        '<think>private reasoning</think>\nmanage: explanation YES\n'
        'visit: No\nresource: tests are appropriate yEs'
    ) == {
        'MANAGE': 'YES',
        'VISIT': 'NO',
        'RESOURCE': 'YES',
        'RESOURCE SPECIFICATION': 'N/A',
    }
    assert majority_vote(['YES', 'NO', None]) == 'YES'
    assert majority_vote([None, 'maybe']) is None


def test_evaluate_majority_metrics_and_unscorable_coverage(tmp_path):
    data_path = tmp_path / 'dataset.csv'
    _write_fixture(data_path)
    dataset = Sen2Med(data_file=str(data_path), validate_release=False)
    predictions = pd.DataFrame({
        'index': dataset.data['index'],
        'prediction': [
            'MANAGE: YES\nVISIT: NO\nRESOURCE: YES',
            'manage: yes\nvisit: no\nresource: yes',
            'not parseable',
            'MANAGE: NO\nVISIT: YES\nRESOURCE: NO',
            'MANAGE: YES\nVISIT: YES\nRESOURCE: YES',
            'not parseable',
        ],
    })
    prediction_path = tmp_path / 'predictions.csv'
    predictions.to_csv(prediction_path, index=False)

    # Native inference mutates dataset.data before the runner calls evaluate.
    dataset.data['prediction'] = 'TRANSIENT NATIVE VALUE'
    metrics = dataset.evaluate(str(prediction_path))

    assert metrics['Accuracy (%) / MANAGE'] == 50.0
    assert metrics['Balanced Accuracy / MANAGE'] == 0.5
    assert metrics['Accuracy (%) / VISIT'] == 100.0
    assert metrics['Balanced Accuracy / VISIT'] == 1.0
    assert metrics['Accuracy (%) / RESOURCE'] == 50.0
    assert metrics['Balanced Accuracy / RESOURCE'] == 0.5
    assert metrics['Unscorable Samples / MANAGE'] == 0
    assert _binary_metrics(['YES'], ['YES'])['balanced_accuracy'] is None


def test_controlled_metrics_filter_by_clinicians_and_score_against_baseline_gold(tmp_path):
    rows = []
    for context_id, perturbation, manage_raters in (
        ('stable', 'baseline', [1.0, 1.0, 1.0, '', '']),
        ('stable', 'gender_swap', [1.0, 1.0, 1.0, '', '']),
        ('changed', 'baseline', [1.0, 1.0, 1.0, '', '']),
        ('changed', 'gender_swap', [0.0, 0.0, 0.0, '', '']),
        ('two-rater-tie', 'baseline', [1.0, 0.0, '', '', '']),
        ('two-rater-tie', 'gender_swap', [1.0, 1.0, 1.0, '', '']),
        ('four-rater-tie', 'baseline', [1.0, 1.0, 0.0, 0.0, '']),
        ('four-rater-tie', 'gender_swap', [1.0, 1.0, 0.0, 0.0, '']),
    ):
        # The retained stable pair deliberately disagrees with the clinician
        # majority, so scoring against that majority would reverse its result.
        manage_gold = 'NO' if context_id == 'stable' else 'YES'
        row = {
            'context_id': context_id,
            'perturbation': perturbation,
            'dataset': 'askdocs',
            'clinical_context': f'{context_id} {perturbation}',
            'gold_manage': manage_gold,
            'gold_visit': 'NO',
            'gold_resource': 'YES',
            **_rater_columns(1.0, 0.0, 1.0),
        }
        for rater, value in enumerate(manage_raters, start=1):
            row[f'clin_rater{rater}_manage'] = value
        rows.append(row)
    data_path = tmp_path / 'controlled.csv'
    pd.DataFrame(rows).to_csv(data_path, index=False)
    dataset = Sen2Med(dataset='Sen2Med', data_file=str(data_path), validate_release=False)

    scenario_predictions = {
        'stable:baseline': 'MANAGE: YES\nVISIT: NO\nRESOURCE: YES',
        'stable:gender_swap': 'MANAGE: NO\nVISIT: NO\nRESOURCE: YES',
        'changed:baseline': 'MANAGE: NO\nVISIT: NO\nRESOURCE: YES',
        'changed:gender_swap': 'MANAGE: YES\nVISIT: NO\nRESOURCE: YES',
        'two-rater-tie:baseline': 'MANAGE: YES\nVISIT: NO\nRESOURCE: YES',
        'two-rater-tie:gender_swap': 'MANAGE: YES\nVISIT: NO\nRESOURCE: YES',
        'four-rater-tie:baseline': 'MANAGE: YES\nVISIT: NO\nRESOURCE: YES',
        'four-rater-tie:gender_swap': 'MANAGE: YES\nVISIT: NO\nRESOURCE: YES',
    }
    predictions = pd.DataFrame({
        'index': dataset.data['index'],
        'prediction': [
            scenario_predictions[index.rsplit(':seed', 1)[0]]
            for index in dataset.data['index']
        ],
    })
    prediction_path = tmp_path / 'controlled_predictions.csv'
    predictions.to_csv(prediction_path, index=False)

    metrics = dataset.evaluate(str(prediction_path))

    # Clinician-majority agreement retains only the stable pair. The changed
    # pair disagrees and exact two- and four-rater ties have no majority.
    assert clinician_majority_vote([1.0, 0.0]) is None
    assert clinician_majority_vote([1.0, 1.0, 0.0, 0.0]) is None
    assert metrics['Controlled Retained Samples / gender_swap / MANAGE'] == 1
    assert metrics['Controlled Baseline Accuracy (%) / gender_swap / MANAGE'] == 0.0
    assert metrics['Controlled Accuracy (%) / gender_swap / MANAGE'] == 100.0
    assert metrics['Controlled Accuracy Change (pp) / gender_swap / MANAGE'] == 100.0


def test_missing_and_malformed_dataset_fail_clearly(tmp_path):
    missing = tmp_path / 'missing.csv'
    try:
        Sen2Med(data_file=str(missing), validate_release=False)
    except FileNotFoundError as error:
        assert 'dataset file is missing' in str(error)
    else:
        raise AssertionError('missing dataset must fail')

    malformed = tmp_path / 'malformed.csv'
    pd.DataFrame([{'context_id': 'case-a'}]).to_csv(malformed, index=False)
    try:
        Sen2Med(data_file=str(malformed), validate_release=False)
    except ValueError as error:
        assert 'missing required columns' in str(error)
    else:
        raise AssertionError('malformed dataset must fail')


class _SchedulerDataset:
    dataset_name = 'SchedulerTest'

    def __init__(self, count):
        self.data = pd.DataFrame({'index': [str(i) for i in range(count)]})

    def __len__(self):
        return len(self.data)

    def build_prompt(self, item):
        return [{'type': 'text', 'value': str(item['index'])}]

    def get_generation_kwargs(self, item):
        return {'seed': int(item['index'])}

    def dump_image(self, line):
        return []


class _BlockingModel:
    is_api = True

    def __init__(self, capacity):
        self.capacity = capacity
        self.release_early = threading.Event()
        self.full = threading.Event()
        self.refilled = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.seeds = []

    def set_dump_image(self, callback):
        self.dump_image = callback

    def generate(self, message, dataset, seed):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.seeds.append(seed)
            if self.active == self.capacity:
                self.full.set()
            if seed >= self.capacity:
                self.refilled.set()
        assert self.full.wait(5)
        if seed == 0:
            assert self.release_early.wait(5)
        with self.lock:
            self.active -= 1
        return json.dumps({'seed': seed})


def test_native_inference_scheduler_forwards_kwargs_and_refills(tmp_path):
    for capacity in (2, 4):
        work_dir = tmp_path / f'workers-{capacity}'
        work_dir.mkdir()
        dataset = _SchedulerDataset(capacity * 2 + 1)
        model = _BlockingModel(capacity)
        result = {}
        error = []

        def run():
            try:
                output = work_dir / 'native-output.pkl'
                infer_data(
                    model,
                    'scheduler-model',
                    str(work_dir),
                    dataset,
                    str(output),
                    api_nproc=capacity,
                    retry_failed=False,
                )
                result.update(load(str(output)))
            except Exception as exc:  # pragma: no cover - asserted below
                error.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        assert model.full.wait(5)
        assert model.refilled.wait(5)
        assert model.peak == capacity
        model.release_early.set()
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert not error
        assert sorted(model.seeds) == list(range(capacity * 2 + 1))
        assert list(result) == list(dataset.data['index'])


def test_openai_payload_has_two_messages_and_seed(monkeypatch, tmp_path):
    data_path = tmp_path / 'dataset.csv'
    _write_fixture(data_path)
    dataset = Sen2Med(data_file=str(data_path), validate_release=False)
    captured = {}

    def post(url, headers, data, proxies, timeout, stream):
        captured.update(json.loads(data))
        return SimpleNamespace(
            status_code=200,
            text=json.dumps({'choices': [{'message': {'content': 'MANAGE: YES'}}]}),
            headers={},
            encoding='utf-8',
            apparent_encoding='utf-8',
        )

    monkeypatch.setattr('vlmeval.api.gpt.requests.post', post)
    monkeypatch.setattr('vlmeval.api.base.rd.random', lambda: 0)
    model = OpenAIWrapper(
        model='gpt-4o',
        api_base='https://example.invalid/v1/chat/completions',
        key='test-key',
        retry=1,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.1,
        repetition_penalty=1.1,
        max_tokens=1024,
        verbose=False,
    )

    answer = model.generate(
        dataset.build_prompt(0),
        dataset=dataset.dataset_name,
        **dataset.get_generation_kwargs(0),
    )

    assert answer == 'MANAGE: YES'
    assert [message['role'] for message in captured['messages']] == ['system', 'user']
    assert captured['messages'][0]['content'] == SYSTEM_PROMPT
    assert captured['temperature'] == 0.1
    assert captured['repetition_penalty'] == 1.1
    assert captured['max_tokens'] == 1024
    assert captured['seed'] == 0
    assert captured['dataset'] == dataset.dataset_name
