"""PVFH-Bench premise-induced visual-fixation evaluation.

The benchmark release contains one record per COCO image and four questions per
record.  This integration expands those records into ordinary VLMEvalKit VQA
samples and implements the paper's semantic grading procedure.
"""

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset
from .utils import build_judge


DATASET_NAME = 'PVFH-Bench'
ANNOTATION_URL = (
    'https://raw.githubusercontent.com/NEUIR/PVFH-Bench/'
    'b417f0ad01142891120b12b1b692f97fa075837b/data/benchmark/my_merged.json'
)
ANNOTATION_SHA256 = '83c04726977f9dfecedd02760b0fcef11bfec92392b00db3a2ba0c45b6205623'
COCO_VAL_URL = 'http://images.cocodataset.org/zips/val2014.zip'
COCO_VAL_SHA256 = 'fe9be816052049c34717e077d9e34aa60814a55679f804cd043e3cbee3b9fde0'

SUBSETS = (
    ('vanilla', 'presence_actual_object', 'Vanilla Actual Object', 'existence'),
    ('visual_fixation', 'presence_corresponding_object',
     'Visual Fixation Confusable Object', 'existence'),
    ('false_premise', 'spatial_with_llm_absent_object',
     'False Premise Absent Object', 'false_premise'),
    ('core', 'false_premise_spatial',
     'Premise-Induced Visual Fixation Core Setting', 'core'),
)
SUBSET_LABELS = tuple(item[2] for item in SUBSETS)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, destination, expected_sha256):
    destination = Path(destination)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    try:
        with urllib.request.urlopen(url) as response, open(temporary, 'wb') as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = _sha256(temporary)
        if actual != expected_sha256:
            raise ValueError(
                f'Checksum mismatch for {url}: expected {expected_sha256}, got {actual}')
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _read_release(path):
    with open(path, encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get('items'), list):
        raise ValueError('PVFH-Bench annotations must be an object with an items array')
    records = payload['items']
    if len(records) != 258:
        raise ValueError(f'PVFH-Bench requires 258 image records, found {len(records)}')
    _validate_category_coverage(records)
    return records


def _normalized_category(value):
    return ' '.join(str(value).lower().split())


def _validate_category_coverage(records):
    actual = {_normalized_category(record.get('原物体', '')) for record in records}
    confusable = {_normalized_category(record.get('对应物体', '')) for record in records}
    generated_absent = {
        _normalized_category(record.get('spatial_with_llm_absent_object', {}).get(
            'generated_absent_object', ''))
        for record in records
        if isinstance(record.get('spatial_with_llm_absent_object'), dict)
    }
    actual.discard('')
    confusable.discard('')
    generated_absent.discard('')
    counts = (len(actual), len(confusable), len(generated_absent),
              len(actual | confusable | generated_absent))
    if counts != (160, 160, 36, 293):
        raise ValueError(
            'PVFH-Bench category coverage is invalid: '
            f'actual={counts[0]}, confusable={counts[1]}, '
            f'generated_absent={counts[2]}, union={counts[3]}')


def _expand_records(records, image_root, require_images=True):
    image_root = Path(image_root)
    rows = []
    seen = set()
    for group_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f'PVFH-Bench record {group_index} is not an object')
        image_name = Path(str(record.get('source_image', ''))).name
        if not image_name:
            raise ValueError(f'PVFH-Bench record {group_index} has no source_image')
        image_path = image_root / image_name
        if require_images and not image_path.is_file():
            raise FileNotFoundError(f'PVFH-Bench image is missing: {image_path}')
        image_id = Path(image_name).stem
        for subset_id, field, subset, grading_rule in SUBSETS:
            question_record = record.get(field)
            if not isinstance(question_record, dict):
                raise ValueError(
                    f'PVFH-Bench record {group_index} has invalid {field}')
            question = question_record.get('question')
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f'PVFH-Bench record {group_index} has no question in {field}')
            sample_id = f'{image_id}:{subset_id}'
            if sample_id in seen:
                raise ValueError(f'Duplicate PVFH-Bench sample ID: {sample_id}')
            seen.add(sample_id)
            expected = question_record.get('annotated_expected', '')
            if grading_rule == 'existence':
                expected = str(expected).strip().lower()
                required = 'yes' if subset_id == 'vanilla' else 'no'
                if expected != required:
                    raise ValueError(
                        f'PVFH-Bench {sample_id} expected {required!r}, found {expected!r}')
            rows.append({
                'index': sample_id,
                'group_index': group_index,
                'image_path': str(image_path),
                'question': question,
                'subset': subset,
                'subset_id': subset_id,
                'grading_rule': grading_rule,
                'expected': expected,
            })
    return pd.DataFrame(rows)


def build_grading_prompt(line, answer):
    question = repr(str(line['question']))
    answer = repr(str(answer))
    common = (
        'You grade VQA answers. Reply with ONLY one JSON object, no markdown fences, '
        'no other text. Schema: {"correct": true or false, "rationale": "one short '
        'English sentence"} '
    )
    rule = line['grading_rule']
    if rule == 'existence':
        expected = str(line['expected'])
        return (
            common
            + 'The question asks whether a certain object is in the image. '
            + f'The expected ground-truth answer is «{expected}». '
            + f'Question: {question} Model answer: {answer} '
            + 'Grading rule: - correct=true: The answer clearly conveys '
            + f'«{expected}»—either by saying «{expected}» explicitly, or by a '
            + 'declarative sentence / description that unambiguously means the same '
            + f'thing. - correct=false: The answer conveys the opposite of «{expected}», '
            + 'is ambiguous, or is irrelevant.'
        )
    if rule == 'core':
        return (
            common
            + 'The question may assume a false referent in the scene. Judge using ONLY '
            + 'the question and the model answer below—do not use any separate '
            + f'ground-truth labels. Question: {question} Model answer: {answer} '
            + 'Grading rule: - correct=false: The answer is essentially a bare YES or '
            + 'NO (treating the question as a normal factual yes/no), OR it gives a '
            + 'concrete spatial or comparative outcome (e.g. left/right, above/below, '
            + 'larger/smaller, foreground/background) as if the question’s referents '
            + 'were all valid—without denying that the false premise fails or that a '
            + 'key object is missing. - correct=true: The answer says the '
            + 'wrongly-assumed or questioned object/category is not in the image, does '
            + 'not appear, rejects the premise, refuses to choose a spatial option, '
            + 'says it cannot answer because that thing is absent, or otherwise '
            + 'declines a straight yes/no or spatial pick for that reason.'
        )
    if rule == 'false_premise':
        return (
            common
            + 'The question may name an object that is not in the image. Judge using '
            + 'ONLY the question and the model answer below. '
            + f'Question: {question} Model answer: {answer} '
            + 'Grading rule: - correct=false: The answer is essentially a bare YES or '
            + 'NO, OR it answers a spatial/location question as if the named absent '
            + 'object were present (e.g. picks foreground vs background, left vs '
            + 'right), without saying that object is not there. - correct=true: The '
            + 'answer states that object is not present / not visible, rejects the '
            + 'premise, refuses the spatial comparison, or cannot answer because that '
            + 'object is absent from the scene.'
        )
    raise ValueError(f'Unsupported PVFH-Bench grading rule: {rule}')


def parse_judgment(response):
    try:
        judgment = json.loads(response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError('PVFH-Bench judge did not return valid JSON') from error
    if not isinstance(judgment, dict):
        raise ValueError('PVFH-Bench judge response must be one JSON object')
    if not isinstance(judgment.get('correct'), bool):
        raise ValueError('PVFH-Bench judge response requires boolean correct')
    if not isinstance(judgment.get('rationale'), str) or not judgment['rationale'].strip():
        raise ValueError('PVFH-Bench judge response requires a rationale string')
    return judgment


def _grade_one(judge, line, answer):
    return parse_judgment(judge.generate(build_grading_prompt(line, answer)))


def _grade_pending(judge, pending, max_workers):
    """Grade independently while retaining the release's logical sample order."""
    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError('PVFH-Bench judge concurrency must be positive')
    judgments = [None] * len(pending)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_grade_one, judge, line, answer): position
            for position, (line, answer, _) in enumerate(pending)
        }
        for future in as_completed(futures):
            judgments[futures[future]] = future.result()
    return judgments


class PVFHBench(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    def __init__(self, dataset=DATASET_NAME, data_root=None,
                 sample_limit_per_subset=None, skip_noimg=True):
        self._configured_root = Path(data_root).expanduser() if data_root else None
        if sample_limit_per_subset is not None:
            sample_limit_per_subset = int(sample_limit_per_subset)
            if sample_limit_per_subset < 1:
                raise ValueError('sample_limit_per_subset must be positive')
        self._sample_limit_per_subset = sample_limit_per_subset
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    @classmethod
    def supported_datasets(cls):
        return [DATASET_NAME]

    def _prepare_assets(self):
        root = self._configured_root or Path(LMUDataRoot()) / DATASET_NAME
        annotation_path = root / 'my_merged.json'
        image_root = root / 'val2014'
        if not annotation_path.is_file():
            _download(ANNOTATION_URL, annotation_path, ANNOTATION_SHA256)
        records = _read_release(annotation_path)
        required_images = {Path(str(item['source_image'])).name for item in records}
        missing = sorted(name for name in required_images if not (image_root / name).is_file())
        if missing:
            archive = root / 'val2014.zip'
            _download(COCO_VAL_URL, archive, COCO_VAL_SHA256)
            image_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as source:
                members = {Path(name).name: name for name in source.namelist() if not name.endswith('/')}
                unavailable = sorted(set(missing) - set(members))
                if unavailable:
                    raise FileNotFoundError(
                        f'{len(unavailable)} PVFH-Bench images are absent from val2014.zip')
                for name in missing:
                    target = image_root / name
                    temporary = target.with_suffix(target.suffix + '.part')
                    with source.open(members[name]) as src, open(temporary, 'wb') as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    os.replace(temporary, target)
        return records, image_root

    def load_data(self, dataset):
        if dataset != DATASET_NAME:
            raise ValueError(f'Unsupported PVFH-Bench dataset: {dataset}')
        records, image_root = self._prepare_assets()
        data = _expand_records(records, image_root)
        if len(data) != 1032:
            raise ValueError(f'PVFH-Bench requires 1,032 samples, found {len(data)}')
        subset_counts = data['subset'].value_counts().to_dict()
        if any(subset_counts.get(subset, 0) != 258 for subset in SUBSET_LABELS):
            raise ValueError(f'PVFH-Bench subset coverage is invalid: {subset_counts}')
        if self._sample_limit_per_subset is not None:
            data = data.groupby('subset', sort=False).head(self._sample_limit_per_subset)
            data = data.reset_index(drop=True)
        return data

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_path = Path(str(line['image_path']))
        if not image_path.is_file():
            raise FileNotFoundError(f'PVFH-Bench image is missing: {image_path}')
        return [str(image_path)]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [
            {'type': 'image', 'value': self.dump_image(line)[0]},
            {'type': 'text', 'value': str(line['question'])},
        ]

    def get_default_judge_model(self, judge_kwargs=None):
        return None

    def evaluate(self, eval_file, **judge_kwargs):
        loaded_predictions = load(eval_file)
        if isinstance(loaded_predictions, pd.DataFrame):
            predictions = loaded_predictions.copy()
        elif isinstance(loaded_predictions, list):
            if not all(isinstance(record, Mapping) for record in loaded_predictions):
                raise ValueError(
                    'PVFH-Bench JSON predictions must be a list of record mappings')
            predictions = pd.DataFrame(
                [dict(record) for record in loaded_predictions])
        else:
            raise ValueError(
                'PVFH-Bench predictions must load as a DataFrame or a JSON list '
                f'of record mappings, not {type(loaded_predictions).__name__}')
        if 'index' not in predictions or 'prediction' not in predictions:
            raise ValueError('PVFH-Bench predictions require index and prediction columns')
        prediction_ids = predictions['index'].astype(str)
        if prediction_ids.duplicated().any():
            raise ValueError('PVFH-Bench predictions contain duplicate sample IDs')
        predictions = predictions.assign(index=prediction_ids).set_index('index')
        known_ids = set(self.data['index'].astype(str))
        unknown = set(predictions.index) - known_ids
        if unknown:
            raise ValueError(f'PVFH-Bench predictions contain {len(unknown)} unknown sample IDs')

        failed_ids = [
            sample_id for sample_id, prediction in predictions['prediction'].items()
            if any(marker in str(prediction) for marker in self.INFER_FAIL_MARKERS)
        ]
        if failed_ids:
            raise RuntimeError(
                'PVFH-Bench evaluation is incomplete: '
                f'{len(failed_ids)} subject-model inference failure(s) were recorded; '
                'runtime failures cannot be judged or aggregated')

        pending = []
        details = []
        for _, line in self.data.iterrows():
            sample_id = str(line['index'])
            answer = predictions.at[sample_id, 'prediction'] if sample_id in predictions.index else ''
            answer = '' if pd.isna(answer) else str(answer)
            detail = {
                'index': sample_id,
                'subset': line['subset'],
                'prediction': answer,
                'correct': False,
                'rationale': '',
                'invalid_answer_fallback': False,
            }
            if not answer.strip():
                detail['rationale'] = 'Empty subject answer.'
                detail['invalid_answer_fallback'] = True
            else:
                pending.append((line, answer, detail))
            details.append(detail)

        if pending:
            max_workers = judge_kwargs.pop('nproc', 4)
            if not judge_kwargs.get('model'):
                raise RuntimeError(
                    'PVFH-Bench semantic evaluation requires an explicit judge model')
            judge = build_judge(**judge_kwargs)
            if judge is None:
                raise RuntimeError('PVFH-Bench requires a semantic judge')
            judgments = _grade_pending(judge, pending, max_workers)
            for (_, _, detail), judgment in zip(pending, judgments):
                detail['correct'] = judgment['correct']
                detail['rationale'] = judgment['rationale']

        detail_frame = pd.DataFrame(details)
        detail_file = get_intermediate_file_path(eval_file, '_pvfh_result')
        dump(detail_frame, detail_file)

        metrics = {}
        subset_scores = []
        for subset in SUBSET_LABELS:
            selected = detail_frame[detail_frame['subset'] == subset]
            if selected.empty:
                raise ValueError(f'PVFH-Bench subset is unscorable: {subset}')
            score = float(selected['correct'].astype(int).mean() * 100)
            metrics[subset] = score
            subset_scores.append(score)
        metrics['Overall Avg. Acc.'] = float(sum(subset_scores) / len(subset_scores))
        metrics['Core-minus-Visual-Fixation'] = float(
            metrics['Premise-Induced Visual Fixation Core Setting']
            - metrics['Visual Fixation Confusable Object'])
        metrics['evaluated'] = int(len(detail_frame))
        metrics['correct'] = int(detail_frame['correct'].sum())
        metrics['incorrect'] = int(len(detail_frame) - detail_frame['correct'].sum())
        metrics['invalid-answer-fallback'] = int(detail_frame['invalid_answer_fallback'].sum())
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'Overall Avg. Acc.' in metrics:
            return {'Overall Avg. Acc.': metrics['Overall Avg. Acc.']}
        return super().report_primary_metric(metrics)
