import asyncio
import json
import threading
from pathlib import Path

import pandas as pd
import pytest

from run import build_dataset_from_config
from vlmeval.api.gpt import OpenAIWrapper
from vlmeval.dataset import SUPPORTED_DATASETS, PVFHBench
from vlmeval.dataset.pvfh_bench import (
    SUBSET_LABELS,
    _expand_records,
    build_grading_prompt,
    parse_judgment,
)
from vlmeval.inference_api import APIEvalPipeline, DatasetConfig
from vlmeval.smp import load


def _release_record(index):
    common = [f'common-{value}' for value in range(40)]
    actual_only = [f'actual-{value}' for value in range(120)]
    confusable_only = [f'confusable-{value}' for value in range(120)]
    actual_categories = common + actual_only
    confusable_categories = common + confusable_only
    absent_categories = common[:23] + [f'absent-{value}' for value in range(13)]
    return {
        'source_image': f'COCO_val2014_{index:012d}.jpg',
        '对应物体': confusable_categories[index % len(confusable_categories)],
        '原物体': actual_categories[index % len(actual_categories)],
        'presence_corresponding_object': {
            'question': f'Is confusable object {index} present?',
            'dataset_kind': 'presence_corresponding_object',
            'annotated_expected': 'No',
        },
        'presence_actual_object': {
            'question': f'Is actual object {index} present?',
            'dataset_kind': 'presence_actual_object',
            'annotated_expected': 'Yes',
        },
        'false_premise_spatial': {
            'anchor_object': 'anchor',
            'false_premise': 'false premise',
            'relation_type': 'left_or_right',
            'question': f'Is confusable object {index} left of the anchor?',
        },
        'spatial_with_llm_absent_object': {
            'anchor_object': 'anchor',
            'generated_absent_object': absent_categories[index % len(absent_categories)],
            'absent_generation_error': None,
            'dataset_kind': 'spatial_with_absent_object',
            'relation_type': 'foreground_or_background',
            'question': f'Is absent object {index} in the foreground?',
            'false_premise': 'false premise',
        },
    }


def _write_release(root, count=258, create_images=True):
    root = Path(root)
    image_root = root / 'val2014'
    image_root.mkdir(parents=True)
    records = [_release_record(index) for index in range(count)]
    (root / 'my_merged.json').write_text(
        json.dumps({'meta': {'item_count': count}, 'items': records}),
        encoding='utf-8',
    )
    if create_images:
        for record in records:
            (image_root / record['source_image']).touch()
    return records


def _evaluation_dataset():
    rows = []
    rules = ('existence', 'existence', 'false_premise', 'core')
    expected = ('yes', 'no', '', '')
    for repeat in range(2):
        for position, (subset, rule, answer) in enumerate(
                zip(SUBSET_LABELS, rules, expected)):
            rows.append({
                'index': f'{repeat}:{position}',
                'question': f'question {repeat}:{position}',
                'subset': subset,
                'grading_rule': rule,
                'expected': answer,
            })
    dataset = PVFHBench.__new__(PVFHBench)
    dataset.dataset_name = 'PVFH-Bench'
    dataset.data = pd.DataFrame(rows)
    return dataset


def test_native_config_constructs_registered_release(tmp_path):
    _write_release(tmp_path)
    config_path = Path(__file__).parents[1] / 'configs' / 'pvfh_bench.json'
    config = load(str(config_path))
    config['PVFH-Bench']['data_root'] = str(tmp_path)

    assert 'PVFH-Bench' in SUPPORTED_DATASETS
    dataset = build_dataset_from_config(config, 'PVFH-Bench', strict=True)
    assert isinstance(dataset, PVFHBench)
    assert len(dataset.data) == 1032
    assert dataset.data['subset'].value_counts().to_dict() == {
        subset: 258 for subset in SUBSET_LABELS
    }
    assert dataset.data.iloc[0]['index'].endswith(':vanilla')
    assert dataset.data.iloc[3]['index'].endswith(':core')
    prompt = dataset.build_prompt(0)
    assert [part['type'] for part in prompt] == ['image', 'text']


def test_native_config_rejects_unknown_option(tmp_path):
    config = {
        'PVFH-Bench': {
            'class': 'PVFHBench',
            'dataset': 'PVFH-Bench',
            'data_root': str(tmp_path),
            'not_a_real_option': True,
        }
    }
    with pytest.raises(ValueError, match='Unsupported parameter'):
        build_dataset_from_config(config, 'PVFH-Bench', strict=True)


def test_release_shape_and_missing_image_are_visible(tmp_path):
    records = _write_release(tmp_path, count=1, create_images=False)
    with pytest.raises(FileNotFoundError, match='image is missing'):
        _expand_records(records, tmp_path / 'val2014')
    records[0]['presence_actual_object'].pop('question')
    with pytest.raises(ValueError, match='has no question'):
        _expand_records(records, tmp_path / 'val2014', require_images=False)


def test_judge_prompt_and_strict_json_contract():
    line = {
        'question': 'Is the premise valid?',
        'expected': 'yes',
        'grading_rule': 'existence',
    }
    prompt = build_grading_prompt(line, 'It is present.')
    assert 'expected ground-truth answer is «yes»' in prompt
    assert parse_judgment('{"correct": true, "rationale": "It agrees."}') == {
        'correct': True,
        'rationale': 'It agrees.',
    }
    with pytest.raises(ValueError, match='valid JSON'):
        parse_judgment('```json\n{"correct": true}\n```')
    with pytest.raises(ValueError, match='boolean correct'):
        parse_judgment('{"correct": 1, "rationale": "Not boolean."}')


def test_invalid_answers_stay_in_every_denominator(tmp_path):
    dataset = _evaluation_dataset()
    predictions = dataset.data[['index']].copy()
    predictions['prediction'] = ''
    eval_file = tmp_path / 'predictions.csv'
    predictions.to_csv(eval_file, index=False)

    metrics = dataset.evaluate(str(eval_file))
    assert {metrics[key] for key in SUBSET_LABELS} == {0.0}
    assert metrics['Overall Avg. Acc.'] == 0.0
    assert metrics['evaluated'] == 8
    assert metrics['incorrect'] == 8
    assert metrics['invalid-answer-fallback'] == 8


def test_nonempty_answers_require_explicit_judge(tmp_path):
    dataset = _evaluation_dataset()
    predictions = dataset.data[['index']].copy()
    predictions['prediction'] = 'an answer'
    eval_file = tmp_path / 'predictions.csv'
    predictions.to_csv(eval_file, index=False)
    with pytest.raises(RuntimeError, match='explicit judge model'):
        dataset.evaluate(str(eval_file))


def test_inference_failure_marker_aborts_before_judging(monkeypatch, tmp_path):
    dataset = _evaluation_dataset()
    predictions = dataset.data[['index']].copy()
    predictions['prediction'] = 'an answer'
    predictions.loc[0, 'prediction'] = (
        f'prefix {dataset.INFER_FAIL_MARKERS[0]}: upstream timeout')
    eval_file = tmp_path / 'predictions.csv'
    predictions.to_csv(eval_file, index=False)

    def unexpected_judge(**kwargs):
        pytest.fail('inference failures must be rejected before judge construction')

    monkeypatch.setattr('vlmeval.dataset.pvfh_bench.build_judge', unexpected_judge)
    with pytest.raises(RuntimeError, match='evaluation is incomplete'):
        dataset.evaluate(str(eval_file), model='judge-under-test')


@pytest.mark.parametrize('capacity', [2, 4])
def test_native_model_pipeline_refills_configured_capacity(monkeypatch, tmp_path, capacity):
    dataset = _evaluation_dataset()
    image_path = tmp_path / 'image.jpg'
    image_path.touch()
    dataset.data['image_path'] = str(image_path)
    expected_ids = list(dataset.data['index'])

    class FakeJudge:
        def generate(self, prompt):
            assert 'Grading rule:' in prompt
            return '{"correct": true, "rationale": "Semantically correct."}'

    monkeypatch.setenv('PRED_FORMAT', 'json')
    monkeypatch.setattr(
        'vlmeval.dataset.pvfh_bench.build_judge',
        lambda **kwargs: FakeJudge(),
    )

    lock = threading.Lock()
    release_first = threading.Event()
    initial_capacity_reached = threading.Event()
    enough_started = threading.Event()
    state = {'active': 0, 'peak': 0, 'started': 0}

    class FakeModel:
        def generate(self, message, dataset=None):
            assert [part['type'] for part in message] == ['image', 'text']
            assert dataset == 'PVFH-Bench'
            with lock:
                call_number = state['started']
                state['started'] += 1
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
                if state['started'] >= capacity:
                    initial_capacity_reached.set()
                if state['started'] >= capacity + 1:
                    enough_started.set()
            if call_number == 0:
                assert release_first.wait(timeout=5)
            elif call_number < capacity:
                assert initial_capacity_reached.wait(timeout=5)
            with lock:
                state['active'] -= 1
            return f'answer-{call_number}'

    work_dir = tmp_path / f'pipeline-{capacity}'
    config = DatasetConfig(
        dataset_name='PVFH-Bench',
        dataset_obj=dataset,
        model_name='model-under-test',
        model_obj=FakeModel(),
        work_dir=str(work_dir),
        result_file=str(work_dir / 'predictions.json'),
        judge_kwargs={'model': 'judge-under-test', 'nproc': capacity},
    )
    pipeline = APIEvalPipeline(
        [config],
        concurrency=capacity,
        monitor_interval=1,
        run_eval=True,
        debug=True,
    )
    errors = []

    def run_pipeline():
        try:
            asyncio.run(pipeline.run())
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    worker = threading.Thread(target=run_pipeline)
    worker.start()
    try:
        assert enough_started.wait(timeout=5), 'native model queue did not refill a free slot'
    finally:
        release_first.set()
    worker.join(timeout=8)

    assert not worker.is_alive()
    assert errors == []
    assert state['peak'] == capacity
    predictions = load(str(work_dir / 'predictions.json'))
    assert [str(row['index']) for row in predictions] == expected_ids
    details = load(str(work_dir / 'predictions_pvfh_result.json'))
    assert [str(row['index']) for row in details] == expected_ids
    status = load(str(work_dir / 'status.json'))['datasets']['PVFH-Bench']
    assert status['status'] == 'done'
    assert status.get('error_message') is None
    metrics = status['metrics']
    assert [metrics[subset] for subset in SUBSET_LABELS] == [100.0] * 4
    assert metrics['Overall Avg. Acc.'] == 100.0
    assert metrics['evaluated'] == len(expected_ids)
    assert metrics['correct'] == len(expected_ids)
    assert metrics['incorrect'] == 0
    assert metrics['invalid-answer-fallback'] == 0


@pytest.mark.parametrize('capacity', [2, 4])
def test_native_evaluator_refills_judge_capacity(monkeypatch, tmp_path, capacity):
    dataset = _evaluation_dataset()
    predictions = dataset.data[['index']].copy()
    predictions['prediction'] = 'substantive answer'
    eval_file = tmp_path / f'predictions-{capacity}.csv'
    predictions.to_csv(eval_file, index=False)

    lock = threading.Lock()
    release_first = threading.Event()
    initial_capacity_reached = threading.Event()
    enough_started = threading.Event()
    state = {'active': 0, 'peak': 0, 'started': 0}

    def fake_generate_inner(self, inputs, **kwargs):
        with lock:
            call_number = state['started']
            state['started'] += 1
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
            if state['started'] >= capacity:
                initial_capacity_reached.set()
            if state['started'] >= capacity + 1:
                enough_started.set()
        if call_number == 0:
            assert release_first.wait(timeout=5)
        elif call_number < capacity:
            assert initial_capacity_reached.wait(timeout=5)
        with lock:
            state['active'] -= 1
        return 0, '{"correct": true, "rationale": "Semantic match."}', None

    monkeypatch.setattr(OpenAIWrapper, 'generate_inner', fake_generate_inner)
    monkeypatch.setattr('vlmeval.api.base.rd.random', lambda: 0)
    errors = []

    def run_evaluation():
        try:
            dataset.evaluate(
                str(eval_file),
                model='judge-under-test',
                api_base='http://judge.invalid/chat/completions',
                key='test-key',
                retry=1,
                nproc=capacity,
            )
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    worker = threading.Thread(target=run_evaluation)
    worker.start()
    try:
        assert enough_started.wait(timeout=5), 'queued judgment did not refill a free slot'
    finally:
        release_first.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert errors == []
    assert state['peak'] == capacity
