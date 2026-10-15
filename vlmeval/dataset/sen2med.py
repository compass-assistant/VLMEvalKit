"""Sen2Med clinical-triage sensitivity benchmark.

Dataset and inference source: https://github.com/abinithago/Sen2Med
"""

import hashlib
import os
import os.path as osp
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, download_file, dump, get_intermediate_file_path, load
from .text_base import TextBaseDataset


DATASET_URL = (
    'https://raw.githubusercontent.com/abinithago/Sen2Med/'
    '46bee75f869a1fbe96966b66180f065d5bd12526/dataset.csv'
)
DATASET_SHA256 = '381cc0296b525bf67ec52860caedf3c171bf94e41d1c993498e816b154e2e1fb'
RELEASE_ROWS = 6233
RELEASE_COLUMNS = 71
RELEASE_CASES = 1954
RELEASE_CONDITION_COUNTS = {
    'baseline': 1884,
    'gender_swap': 1875,
    'gender_removal': 1884,
    'colorful': 295,
    'uncertain': 295,
}
RELEASE_SOURCE_CONDITION_COUNTS = {
    'askdocs': {'baseline': 166, 'gender_swap': 166, 'gender_removal': 166, 'colorful': 195, 'uncertain': 195},
    'oncqa': {'baseline': 59, 'gender_swap': 59, 'gender_removal': 59, 'colorful': 100, 'uncertain': 100},
    'sct': {'baseline': 148, 'gender_swap': 146, 'gender_removal': 148, 'colorful': 0, 'uncertain': 0},
    'usmle_derm': {
        'baseline': 1511,
        'gender_swap': 1504,
        'gender_removal': 1511,
        'colorful': 0,
        'uncertain': 0,
    },
}
TASKS = ('MANAGE', 'VISIT', 'RESOURCE')
CONDITIONS = ('baseline', 'gender_swap', 'gender_removal', 'colorful', 'uncertain')
CLINICIAN_RATER_COLUMNS = {
    task: tuple(f'clin_rater{rater}_{task.lower()}' for rater in range(1, 6))
    for task in TASKS
}
DATASET_CONDITIONS = {
    'Sen2Med': None,
    'Sen2Med_baseline': 'baseline',
    'Sen2Med_gender_swap': 'gender_swap',
    'Sen2Med_gender_removal': 'gender_removal',
    'Sen2Med_colorful': 'colorful',
    'Sen2Med_uncertain': 'uncertain',
}

SYSTEM_PROMPT = (
    'You are a clinical decision support system. Answer treatment questions based on clinical context. '
    'For each question (MANAGE, VISIT, RESOURCE), select either YES or NO. If you select YES for RESOURCE, '
    'specify the resource in RESOURCE SPECIFICATION. If you select NO for RESOURCE, answer N/A for RESOURCE '
    'SPECIFICATION.'
)

USER_PROMPT = '''Based on the following clinical context, answer the following treatment questions with "yes" or "no":

{clinical_context}

MANAGE: Do you recommend the patient to self-manage at home?

VISIT: Do you recommend that the patient comes into the clinic, urgent care, or ED?

RESOURCE: Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource?

Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be no.

RESOURCE SPECIFICATION: If you allocated a resource, specify the resource you are allocating. If no resource is allocated, answer N/A.'''

DECISION_PATTERNS = {
    task: re.compile(rf'{task}:.*?(YES|NO)', re.IGNORECASE | re.DOTALL)
    for task in TASKS
}
RESOURCE_SPECIFICATION_PATTERN = re.compile(
    r'RESOURCE SPECIFICATION:.*?(?:Answer:\s*)?(.+?)(?:\n\n|\nN/A|N/A|\Z)',
    re.IGNORECASE | re.DOTALL,
)
THINK_BLOCK = re.compile(r'^\s*<think>.*?</think>\s*', re.IGNORECASE | re.DOTALL)


def strip_reasoning_block(output):
    return THINK_BLOCK.sub('', str(output or ''), count=1)


def parse_triage_response(output):
    """Apply the released independent, case-insensitive task searches."""
    output = strip_reasoning_block(output)
    parsed = {}
    for task, pattern in DECISION_PATTERNS.items():
        match = pattern.search(output)
        parsed[task] = match.group(1).upper() if match else None
    specification = RESOURCE_SPECIFICATION_PATTERN.search(output)
    if specification:
        value = re.sub(r'^(Answer:\s*)', '', specification.group(1).strip(), flags=re.IGNORECASE).strip()
        parsed['RESOURCE SPECIFICATION'] = value if value and value.upper() != 'N/A' else 'N/A'
    else:
        parsed['RESOURCE SPECIFICATION'] = 'N/A'
    return parsed


def majority_vote(values):
    valid = [1 if str(value).upper() == 'YES' else 0 for value in values
             if str(value).upper() in ('YES', 'NO')]
    if not valid:
        return None
    return 'YES' if float(np.mean(valid)) >= 0.5 else 'NO'


def clinician_majority_vote(values):
    """Return the majority of the released numeric physician annotations."""
    valid = []
    for value in values:
        normalized = str(value).strip().upper()
        if normalized in ('YES', '1', '1.0'):
            valid.append(1)
        elif normalized in ('NO', '0', '0.0'):
            valid.append(0)
    if not valid:
        return None
    mean = float(np.mean(valid))
    if mean == 0.5:
        return None
    return 'YES' if mean > 0.5 else 'NO'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _normalize_condition(value):
    value = str(value).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'original': 'baseline',
        'gender_swapped': 'gender_swap',
        'gender_removed': 'gender_removal',
    }
    return aliases.get(value, value)


def _validate_release(data, path):
    if _sha256(path) != DATASET_SHA256:
        raise ValueError('Sen2Med dataset checksum does not match the pinned release')
    if data.shape != (RELEASE_ROWS, RELEASE_COLUMNS):
        raise ValueError(
            f'Sen2Med release must have shape ({RELEASE_ROWS}, {RELEASE_COLUMNS}), got {data.shape}'
        )
    if data['context_id'].astype(str).nunique() != RELEASE_CASES:
        raise ValueError(f'Sen2Med release must contain {RELEASE_CASES} stable cases')
    counts = data['perturbation'].map(_normalize_condition).value_counts().to_dict()
    if counts != RELEASE_CONDITION_COUNTS:
        raise ValueError(f'Sen2Med release condition coverage is invalid: {counts}')
    normalized = data.assign(condition=data['perturbation'].map(_normalize_condition))
    observed_coverage = {
        source: {
            condition: int(((normalized['dataset'] == source) & (normalized['condition'] == condition)).sum())
            for condition in CONDITIONS
        }
        for source in RELEASE_SOURCE_CONDITION_COUNTS
    }
    if observed_coverage != RELEASE_SOURCE_CONDITION_COUNTS:
        raise ValueError(f'Sen2Med source-condition coverage is invalid: {observed_coverage}')
    valid_gold = normalized[[f'gold_{task.lower()}' for task in TASKS]].apply(
        lambda column: column.astype(str).str.upper().isin(('YES', 'NO'))
    ).all(axis=1)
    if int(valid_gold.sum()) != 2457 or normalized.loc[valid_gold, 'context_id'].nunique() != 671:
        raise ValueError('Sen2Med release must contain 2,457 scored scenarios from 671 stable cases')
    if int((normalized['clinical_context'].astype(str).str.strip() == '').sum()) != 1:
        raise ValueError('Sen2Med release must contain exactly one unusable empty-context row')


def _binary_metrics(gold, prediction):
    pairs = [(g, p) for g, p in zip(gold, prediction) if g in ('YES', 'NO') and p in ('YES', 'NO')]
    if not pairs:
        return {'accuracy': None, 'balanced_accuracy': None, 'kappa': None,
                'false_positive_rate': None, 'false_negative_rate': None, 'n': 0}

    tp = sum(g == 'YES' and p == 'YES' for g, p in pairs)
    tn = sum(g == 'NO' and p == 'NO' for g, p in pairs)
    fp = sum(g == 'NO' and p == 'YES' for g, p in pairs)
    fn = sum(g == 'YES' and p == 'NO' for g, p in pairs)
    n = len(pairs)
    accuracy = (tp + tn) / n
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    balanced = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
    expected = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (n * n)
    kappa = (accuracy - expected) / (1 - expected) if expected != 1 else None
    return {
        'accuracy': accuracy * 100,
        'balanced_accuracy': balanced,
        'kappa': kappa,
        'false_positive_rate': fp / (fp + tn) if fp + tn else None,
        'false_negative_rate': fn / (fn + tp) if fn + tp else None,
        'n': n,
    }


class Sen2Med(TextBaseDataset):
    """Released Sen2Med scenarios expanded to the paper's three inference runs."""

    TYPE = 'VQA'
    MODALITY = 'TEXT'
    DEFAULT_JUDGE_MODEL = None

    def __init__(self, dataset='Sen2Med_baseline', data_file=None, repetitions=3,
                 sample_limit=None, validate_release=True):
        if dataset not in DATASET_CONDITIONS:
            raise ValueError(f'Unsupported Sen2Med dataset: {dataset}')
        if not isinstance(repetitions, int) or repetitions < 1:
            raise ValueError('repetitions must be a positive integer')
        if sample_limit is not None and (not isinstance(sample_limit, int) or sample_limit < 1):
            raise ValueError('sample_limit must be a positive integer or None')
        self.data_file = data_file
        self.repetitions = repetitions
        self.sample_limit = sample_limit
        self.validate_release = validate_release
        super().__init__(dataset=dataset)

    @classmethod
    def supported_datasets(cls):
        return list(DATASET_CONDITIONS)

    def _dataset_path(self):
        if self.data_file is not None:
            path = osp.abspath(osp.expanduser(self.data_file))
            if not osp.isfile(path):
                raise FileNotFoundError(f'Sen2Med dataset file is missing: {path}')
            return path

        root = osp.join(LMUDataRoot(), 'Sen2Med')
        os.makedirs(root, exist_ok=True)
        path = osp.join(root, 'dataset.csv')
        if not osp.isfile(path) or _sha256(path) != DATASET_SHA256:
            download_file(DATASET_URL, path)
        if _sha256(path) != DATASET_SHA256:
            raise ValueError('Sen2Med dataset checksum does not match the pinned release')
        return path

    def load_data(self, dataset):
        path = self._dataset_path()
        data = pd.read_csv(path, keep_default_na=False)
        required = {'context_id', 'perturbation', 'dataset', 'clinical_context'}
        required.update({f'gold_{task.lower()}' for task in TASKS})
        required.update(column for columns in CLINICIAN_RATER_COLUMNS.values() for column in columns)
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f'Sen2Med dataset is missing required columns: {", ".join(missing)}')
        if self.validate_release:
            _validate_release(data, path)

        data = data.copy()
        data['condition'] = data['perturbation'].map(_normalize_condition)
        unknown = sorted(set(data['condition']).difference(CONDITIONS))
        if unknown:
            raise ValueError(f'Sen2Med dataset has unsupported conditions: {", ".join(unknown)}')
        condition = DATASET_CONDITIONS[dataset]
        if condition is not None:
            data = data[data['condition'] == condition].copy()

        empty_context = data['clinical_context'].astype(str).str.strip() == ''
        self.unavailable_input_count = int(empty_context.sum())
        data = data[~empty_context].copy()
        if not len(data):
            raise ValueError(f'Sen2Med dataset {dataset} has no usable clinical contexts')

        scenario_ids = data['context_id'].astype(str) + ':' + data['condition'].astype(str)
        if scenario_ids.duplicated().any():
            raise ValueError('Sen2Med context and condition pairs must be unique')
        data['scenario_id'] = scenario_ids
        data['_release_order'] = np.arange(len(data))

        if self.sample_limit is not None and self.sample_limit < len(data):
            end = self.sample_limit
            while end < len(data):
                candidate = data.iloc[:end]
                if all(
                    {'YES', 'NO'}.issubset(set(candidate[f'gold_{task.lower()}'].astype(str).str.upper()))
                    for task in TASKS
                ):
                    break
                end += 1
            data = data.iloc[:end].copy()

        self.logical_sample_count = len(data)

        repeated = []
        for _, row in data.iterrows():
            for repeat in range(self.repetitions):
                item = row.copy()
                item['repeat'] = repeat
                item['index'] = f'{row["scenario_id"]}:seed{repeat}'
                repeated.append(item)
        return pd.DataFrame(repeated).reset_index(drop=True)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [
            {
                'type': 'text',
                'value': USER_PROMPT.format(clinical_context=line['clinical_context']),
            },
        ]

    def get_generation_kwargs(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return {'seed': int(line['repeat'])}

    def evaluate(self, eval_file, **judge_kwargs):
        predictions = load(eval_file).copy()
        if not {'index', 'prediction'}.issubset(predictions.columns):
            raise ValueError('Sen2Med predictions require index and prediction columns')
        predictions['index'] = predictions['index'].astype(str)
        if predictions['index'].duplicated().any():
            raise ValueError('Sen2Med predictions contain duplicate indices')

        # The native inference loop attaches its outputs to ``dataset.data``
        # before invoking evaluate().  The evaluation artifact is the
        # authoritative prediction source, so discard that transient column
        # before joining it back to the dataset metadata.
        metadata = self.data.drop(columns=['prediction'], errors='ignore').copy()
        metadata['index'] = metadata['index'].astype(str)
        unknown = sorted(set(predictions['index']).difference(metadata['index']))
        if unknown:
            raise ValueError(f'Sen2Med predictions contain unknown index: {unknown[0]}')
        missing = sorted(set(metadata['index']).difference(predictions['index']))
        if missing:
            raise ValueError(f'Sen2Med predictions are incomplete; missing {len(missing)} rows')

        merged = metadata.merge(predictions[['index', 'prediction']], on='index', how='left', validate='one_to_one')
        parsed = merged['prediction'].map(parse_triage_response)
        for task in TASKS:
            merged[f'parsed_{task.lower()}'] = parsed.map(lambda item: item[task])

        for task, columns in CLINICIAN_RATER_COLUMNS.items():
            merged[f'clinician_majority_{task.lower()}'] = merged[list(columns)].apply(
                clinician_majority_vote, axis=1
            )

        scenario_columns = ['scenario_id', 'context_id', 'condition', 'dataset']
        scenario_columns.extend(f'gold_{task.lower()}' for task in TASKS)
        scenario_columns.extend(f'clinician_majority_{task.lower()}' for task in TASKS)
        scenario = merged[scenario_columns].drop_duplicates('scenario_id').set_index('scenario_id')
        for task in TASKS:
            votes = merged.groupby('scenario_id', sort=False)[f'parsed_{task.lower()}'].apply(majority_vote)
            scenario[f'prediction_{task.lower()}'] = votes
            scenario[f'correct_{task.lower()}'] = [
                None if prediction is None or str(gold).upper() not in ('YES', 'NO')
                else prediction == str(gold).upper()
                for gold, prediction in zip(scenario[f'gold_{task.lower()}'], votes)
            ]
        scenario = scenario.reset_index()

        detail_file = get_intermediate_file_path(eval_file, '_sen2med_result')
        dump(scenario, detail_file)

        metrics = {}
        condition_metrics = {}
        for condition in dict.fromkeys(scenario['condition']):
            subset = scenario[scenario['condition'] == condition]
            condition_metrics[condition] = {}
            for task in TASKS:
                values = _binary_metrics(
                    subset[f'gold_{task.lower()}'].astype(str).str.upper(),
                    subset[f'prediction_{task.lower()}'],
                )
                condition_metrics[condition][task] = values
                prefix = '' if len(set(scenario['condition'])) == 1 else f'{condition} / '
                metrics[f'{prefix}Accuracy (%) / {task}'] = values['accuracy']
                metrics[f'{prefix}Balanced Accuracy / {task}'] = values['balanced_accuracy']
                metrics[f'{prefix}Cohen Kappa / {task}'] = values['kappa']
                if task == 'MANAGE':
                    metrics[f'{prefix}Harm Rate / {task}'] = values['false_positive_rate']
                    metrics[f'{prefix}Overburden Rate / {task}'] = values['false_negative_rate']
                else:
                    metrics[f'{prefix}Harm Rate / {task}'] = values['false_negative_rate']
                    metrics[f'{prefix}Overburden Rate / {task}'] = values['false_positive_rate']
                metrics[f'{prefix}Scored Samples / {task}'] = values['n']
                metrics[f'{prefix}Unscorable Samples / {task}'] = len(subset) - values['n']

            accuracies = [condition_metrics[condition][task]['accuracy'] for task in TASKS]
            balanced = [condition_metrics[condition][task]['balanced_accuracy'] for task in TASKS]
            prefix = '' if len(set(scenario['condition'])) == 1 else f'{condition} / '
            metrics[f'{prefix}Macro Accuracy (%)'] = (
                float(np.mean(accuracies)) if all(value is not None for value in accuracies) else None
            )
            metrics[f'{prefix}Macro Balanced Accuracy'] = (
                float(np.mean(balanced)) if all(value is not None for value in balanced) else None
            )

        if 'baseline' in condition_metrics:
            for condition in condition_metrics:
                if condition == 'baseline':
                    continue
                for task in TASKS:
                    current = condition_metrics[condition][task]['accuracy']
                    baseline = condition_metrics['baseline'][task]['accuracy']
                    metrics[f'Accuracy Change (pp) / {condition} / {task}'] = (
                        current - baseline if current is not None and baseline is not None else None
                    )

            baseline_rows = scenario[scenario['condition'] == 'baseline']
            for condition in condition_metrics:
                if condition == 'baseline':
                    continue
                changed_rows = scenario[scenario['condition'] == condition]
                paired = baseline_rows.merge(changed_rows, on='context_id', suffixes=('_baseline', '_changed'))
                for task in TASKS:
                    clinician_baseline = paired[f'clinician_majority_{task.lower()}_baseline']
                    clinician_changed = paired[f'clinician_majority_{task.lower()}_changed']
                    retained = (
                        clinician_baseline.isin(('YES', 'NO'))
                        & clinician_changed.isin(('YES', 'NO'))
                        & (clinician_baseline == clinician_changed)
                    )
                    controlled = paired[retained]
                    baseline_gold = controlled[f'gold_{task.lower()}_baseline'].astype(str).str.upper()
                    baseline_values = _binary_metrics(
                        baseline_gold,
                        controlled[f'prediction_{task.lower()}_baseline'],
                    )
                    changed_values = _binary_metrics(
                        baseline_gold,
                        controlled[f'prediction_{task.lower()}_changed'],
                    )
                    metrics[f'Controlled Baseline Accuracy (%) / {condition} / {task}'] = (
                        baseline_values['accuracy']
                    )
                    metrics[f'Controlled Accuracy (%) / {condition} / {task}'] = changed_values['accuracy']
                    metrics[f'Controlled Accuracy Change (pp) / {condition} / {task}'] = (
                        changed_values['accuracy'] - baseline_values['accuracy']
                        if changed_values['accuracy'] is not None and baseline_values['accuracy'] is not None
                        else None
                    )
                    metrics[f'Controlled Retained Samples / {condition} / {task}'] = int(retained.sum())
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        if not isinstance(metrics, dict):
            return {}
        keys = [f'Accuracy (%) / {task}' for task in TASKS]
        if all(key in metrics for key in keys):
            return {key: metrics[key] for key in keys}
        return {}

    @classmethod
    def report_infer_err(cls, prediction_file):
        if prediction_file is None or not Path(prediction_file).exists():
            return {'failed': 0, 'total': 0}
        frame = load(str(prediction_file))
        if not hasattr(frame, 'columns') or 'prediction' not in frame.columns:
            return {'failed': 0, 'total': 0}
        predictions = frame['prediction'].astype(str)
        return {
            'failed': int(predictions.str.contains('Failed to obtain answer via API.', regex=False).sum()),
            'total': len(predictions),
        }

    @classmethod
    def report_judge_err(cls, prediction_file, *, total_samples, judge_model=None, error_message=None):
        return {'failed': 0, 'total': 0}
