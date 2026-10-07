"""N01: derive aggregate quality evidence without model evaluation or text exports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_dataset import audit, sha256_file


def words(text: str) -> list[str]:
    text = unicodedata.normalize('NFC', text).casefold()
    # Punctuation separates words; ё and е remain different, numbers stay literal.
    return ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split()


def word_error_rate(reference: str, hypothesis: str) -> dict:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        raise ValueError('Reference must contain words')
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    errors = previous[-1]
    return {'word_edits': errors, 'reference_words': len(ref), 'hypothesis_words': len(hyp),
            'wer': errors / len(ref)}


def read(path: str):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def ratio(n: int, d: int) -> dict:
    return {'count': n, 'denominator': d, 'percent': 100 * n / d}


def build(reference_path: Path | None = None) -> dict:
    source = read('configs/dataset_source.json')
    current, _ = audit(ROOT, source)  # Checks hashes, schema and basic fields; no model predictions.
    prep = read('reports/data_prep/audit.json')
    if prep['status'] != 'completed':
        raise ValueError('T03 audit is incomplete')
    for split, info in current['splits'].items():
        if (info['sha256'] != prep['source_files'][split]['sha256_after']
                or info['rows'] != prep['source_summary'][split]['rows']
                or info['language_counts']['ru'] != prep['ru_summary_before_cleaning'][split]['rows']):
            raise ValueError(f'T02/T03 evidence differs: {split}')
    total = sum(x['rows'] for x in current['splits'].values())
    ru = sum(x['rows'] for x in prep['ru_summary_before_cleaning'].values())
    kept = sum(x['rows'] for x in prep['final_summary'].values())
    removed = prep['excluded_rows']
    exact_removed = sum(prep['exact_dedup'][key] for key in (
        'removed_exact_train_vs_eval', 'removed_exact_validation_vs_test', 'removed_exact_within_train', 'removed_exact_within_validation'))
    if ru - kept != removed or removed != exact_removed + prep['near_duplicates']['rows_removed']:
        raise ValueError('Exclusion counts do not reconcile')
    speech = read('reports/speech/verification.json')
    hypothesis = read('artifacts/speech/transcripts.json')['voice_warm']['text']
    audio_path = ROOT / 'artifacts/speech/teacher_debug.ogg'
    audio_hash = sha256_file(audio_path)
    hyp_hash = hashlib.sha256(hypothesis.encode()).hexdigest()
    saved = speech['results']['voice_warm']
    if audio_hash != saved['audio_sha256'] or hyp_hash != saved['text_sha256']:
        raise ValueError('Local audio/transcript differs from the saved T08 run')
    import av
    with av.open(str(audio_path)) as container:
        stream = container.streams.audio[0]
        duration = sum(frame.samples / frame.sample_rate for frame in container.decode(stream))
        if abs(duration - saved['duration_seconds']) > 0.001:
            raise ValueError('Decoded audio duration differs from the saved run')
        audio = {'codec': stream.codec_context.name, 'sample_rate_hz': stream.codec_context.sample_rate,
                 'channels': stream.codec_context.channels, 'duration_seconds': saved['duration_seconds'],
                 'bytes': audio_path.stat().st_size, 'sha256': audio_hash,
                 'transcript_sha256': hyp_hash, 'saved_run_date': '2026-09-26',
                 'wer_status': 'pending_human_reference', 'wer': None}
    if reference_path is not None:
        reference = json.loads(reference_path.read_text(encoding='utf-8'))
        if (reference.get('human_verified') is not True or not reference.get('reviewer')
                or not reference.get('verified_on') or reference.get('audio_sha256') != audio_hash):
            raise ValueError('Reference needs explicit human confirmation, date, reviewer and matching audio hash')
        audio.update(word_error_rate(reference['text'], hypothesis))
        audio.update(wer_status='human_verified_reference', reference_sha256=hashlib.sha256(reference['text'].encode()).hexdigest(),
                     reference_file_sha256=sha256_file(reference_path), reviewer=reference['reviewer'], verified_on=reference['verified_on'])
    evidence = ['configs/dataset_source.json', 'configs/baseline.json', 'src/prepare_data.py',
                'reports/data_prep/audit.json', 'reports/dataset/agent_review.json',
                'reports/dataset/human_confirmation.json', 'reports/speech/verification.json']
    return {'source_revision': source['revision'], 'source_files_verified': True,
            'source_total': total, 'source_ru': ru, 'prepared_total': kept,
            'source_splits': current['splits'], 'prepared_splits': prep['final_summary'],
            'language_counts': {lang: sum(x['language_counts'][lang] for x in current['splits'].values()) for lang in ('ru','kk','cs')},
            'non_ru_excluded': ratio(total - ru, total),
            'technical_invalid_rows': ratio(0, total),
            'exact_removed': ratio(exact_removed, ru), 'near_removed': ratio(prep['near_duplicates']['rows_removed'], ru),
            'all_duplicates_removed': ratio(removed, ru), 'retained_ru': ratio(kept, ru),
            'semantic_noise_fraction': None, 'guaranteed_clean_fraction': None,
            'near_duplicate_test_rows_preserved': ratio(prep['near_duplicates']['near_test_rows_preserved'], prep['final_summary']['test']['rows']),
            'final_movie_id_overlap': prep['final_movie_id_overlap'], 'final_exact_text_overlap': prep['final_exact_text_overlap'],
            'label_conflict_groups': prep['label_conflicts']['label_conflict_groups'],
            'agent_review': read('reports/dataset/agent_review.json')['decision_counts'],
            'human_confirmation_count': read('reports/dataset/human_confirmation.json')['reviewed_count'],
            'debug_audio': audio, 'model_test_evaluation_performed': False,
            'evidence_sha256': {path: sha256_file(ROOT / path) for path in evidence}}


def render(r: dict) -> str:
    def pct(key):
        v = r[key]
        return f"{v['count']:,} / {v['denominator']:,} = **{v['percent']:.4f}%**".replace(',', ' ')
    rows = []
    for split, s in r['source_splits'].items():
        p = r['prepared_splits'][split]
        rows.append(f"| {split} | {s['rows']} | {s['language_counts']['ru']} | {p['rows']} | {p['negative']} | {p['neutral']} | {p['positive']} |")
    a = r['debug_audio']
    wer = ('**WER не измерен:** нет проверенного человеком эталона. Ответ Whisper не используется как собственный эталон. '
           'Это открытый критерий N01; задача не считается полностью завершённой.')
    if a['wer_status'] == 'human_verified_reference':
        wer = f"**WER = {a['wer']:.4%}**: {a['word_edits']} правок / {a['reference_words']} слов эталона. Эталон проверен: {a['reviewer']}, {a['verified_on']}."
    return f'''# N01 — качество данных и транскрипций

Подготовлено 07.10.2026 для [сдачи NLP](../../docs/nlp_submission.md).
[Числа и контрольные суммы](data_quality.json) воспроизводятся командой ниже.
Это описание данных; accuracy классификатора и разбор его ошибок относятся к N02.

## Покрытие и состав

Корпус **100k Movie Reviews from Kazakhstan**, автор Rustem Yeshpanov (2026),
письменные отзывы о фильмах с kino.kz. Конфигурация `pc`, ревизия
`{r['source_revision']}`. Заявленная лицензия CC BY 4.0 и атрибуция —
в [описании источника](../../NLP_dataset/README.md). Происхождение подтверждено
побайтовым совпадением с закреплённым манифестом; все три SHA-256 перепроверены.
Это корпус киноотзывов, не выгрузка Telegram. Telegram — интерфейс ввода.
Перенос на произвольные темы не проверен. Даты публикации отзывов и демография
авторов не представлены полями корпуса, соответствующее покрытие неизвестно.

Три файла Parquet, четыре строковых поля: `movie_id` — ID фильма;
`review_text` — исходный текст; `review_sentiment` — метка;
`review_language` — язык (`ru`, `kk`, `cs`). Всего **{r['source_total']}** записей:
ru — {r['language_counts']['ru']}, kk — {r['language_counts']['kk']},
cs — {r['language_counts']['cs']}. Для обучения отбирается только ru.
Метки: negative=0, neutral=1, positive=2; рейтинги не переводятся в классы.

| Split | Исходный, все языки | ru до очистки | После очистки | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

После подготовки **{r['prepared_total']}** строк. На train neutral составляет
{100*r['prepared_splits']['train']['neutral']/r['prepared_splits']['train']['rows']:.2f}%,
positive — {100*r['prepared_splits']['train']['positive']/r['prepared_splits']['train']['rows']:.2f}%.
Из-за дисбаланса accuracy нужно сопровождать macro-F1 и метриками каждого класса.

## Предобработка и техническое качество

1. Проверка SHA-256, схемы, числа строк и допустимых значений.
2. Отбор `review_language=ru`. Исключено по языку {pct('non_ru_excluded')}.
   Это ограничение области эксперимента, не шум или плохое качество других языков.
3. Для поиска повторов: Unicode NFC → casefold → удаление невидимых символов
   согласно реализации → свёртка пробелов. Ключ нормализации используется
   для сравнения; исходный `review_text` не заменяется.
4. Удаление точных повторов и подтверждённых близких копий из train/validation;
   приоритет сохранения test > validation > train. Близость — Jaccard по
   символьным триграммам с порогом 0,9, затем сохранённые решения команды.
5. В baseline исходный текст обрабатывается TF-IDF: lowercase, словные 1–2-граммы,
   min_df=2, max_df=0,98, max_features=100000, sublinear_tf, L2-нормировка.
   Нет лемматизации и отдельной ручной нормализации входа модели. Токенизатор
   рассчитан на слова длиной от двух символов; обработка пунктуации/эмодзи
   зависит от него. Фактическая конфигурация — [baseline.json](../../configs/baseline.json);
   межплатформенное расхождение T04 ещё расследуется. Словарь и IDF обучаются
   только на train.

Технически дефектной здесь считается строка с пропущенным, нестроковым или
пустым обязательным полем либо неизвестной меткой/языком. Повторная проверка
исходных файлов: {pct('technical_invalid_rows')}. Это узкое определение,
не оценка орфографии, сарказма, достоверности меток или релевантности содержания.

| Показатель | Числитель / знаменатель и доля |
|---|---|
| Исключено точных повторов | {pct('exact_removed')} |
| Исключено подтверждённых близких копий | {pct('near_removed')} |
| Всего исключено повторов | {pct('all_duplicates_removed')} |
| Сохранено русскоязычных строк | {pct('retained_ru')} |

Знаменатель этих четырёх строк — все ru до очистки. Числа исключений взяты
из завершённого T03, сверены с суммами строк; алгоритм дедупликации здесь
заново не запускался. Подготовка T03 ранее независимо воспроизведена:
[протокол](../acceptance/2026-10-03.md).

**Доля смыслового шума и доля гарантированно чистых записей неизвестны.**
Сохранённые строки прошли перечисленные проверки, но не становятся эталонно
чистыми. Neutral не считается шумом. 58 удалённых строк и 58 подтверждённых
пар близких копий — разные величины; совпадение чисел случайно. Из 59 кандидатов
58 пар признаны копиями; по ним исключены 49 строк, ещё 9 — точные повторы.

Пересечения фильмов и нормализованных точных текстов между подготовленными
сплитами — 0. Конфликтов меток на точных совпадениях — 0. Это не исключает
все возможные смысловые утечки. Поиск близких копий ограничен его блокировкой,
длиной и порогом. В неизменном test сохранены 7 пар близких копий, 14 строк:
{pct('near_duplicate_test_rows_preserved')}. Их влияние на финальные метрики
будет оцениваться позже; модельные предсказания на test сейчас не вычислялись.

## Качество исходной разметки

По описанию источника разметку выполнил один автор; независимое согласие
разметчиков не измерено. Полные правила автора недоступны в сохранённой
проверке. Рабочие правила: преобладающая оценка фильма; neutral — отсутствие
ясного итога либо смешанная оценка без перевеса.

В T02 агент прочитал 100 train-отзывов с видимыми исходными метками:
80 счёл обоснованными, 15 неоднозначными, в 5 предложил другую метку.
Пользователь подтвердил 15 выбранных случаев: 13 согласий, №63 и №94 — neutral.
[Протокол](../dataset/manual_review.md). Это не 100 проверок человеком и не
слепая независимая разметка. Сбалансированная выборка сильно перепредставляет
neutral; проценты из неё не переносим на весь корпус. Исходные метки сохранены.
Основные ограничения — смешанные оценки, сарказм, ожидания до просмотра,
отношение к актёрам/возрастным ограничениям вместо оценки фильма.

## Аудио и качество транскрипций

Для обучающего корпуса письменных отзывов **продолжительность, аудиопараметры
и качество транскрипций неприменимы**: его тексты не получены распознаванием речи.

Отдельная отладочная проверка T08: фрагмент предоставленного пользователем видео
преподавателя, 00:10–00:22. Это речь об устройстве задания, не киноотзыв и не
пример финального голосового теста. Длительность {a['duration_seconds']} с,
OGG/{a['codec']}, {a['sample_rate_hz']} Гц, {a['channels']} канал,
{a['bytes']} байт. Метаданные, SHA аудио и сохранённой транскрипции сверены
с локальными файлами; текста и аудио в Git нет.

Whisper small / faster-whisper, CPU int8 (фактически int8_float32), русский язык,
beam=5, VAD. [Сохранённая проверка 26.09.2026](../speech/verification.md):
первый вызов 61,744 с, повторный 60,080 с, оба вернули одинаковый текст;
тишина → empty, повреждённый OGG → decode_failed. Это проверка работоспособности,
не оценка точности транскрипции. В N01 распознавание заново не запускалось.
Один короткий фрагмент не оценивает шумную речь и всё разнообразие Telegram;
доля чистых/шумных аудиосегментов не размечена и неизвестна.

{wer}

Протокол WER: эталон проверяется человеком по аудио; для эталона и гипотезы
Unicode NFC, casefold, пунктуация заменяется пробелами, разбиение по пробельным
символам. Ё/е различаются, числа не переводятся в слова. WER = минимальное
число вставок, удалений и замен слов / число слов эталона; может превышать 100%.
Пустой эталон отклоняется. Обрезанные границы фрагмента и подсказка уже видимого
ответа ASR ограничивают независимость проверки; даже нулевой WER на этом
фрагменте не доказывает качество на других записях.

## Воспроизведение

```bash
python tools/report_nlp_data_quality.py
# После явной проверки эталона человеком:
python tools/report_nlp_data_quality.py --reference artifacts/speech/human_reference.json
```

Локальный JSON эталона содержит `text`, `audio_sha256`, `human_verified: true`,
`reviewer`, `verified_on`. Флаг фиксирует реальное подтверждение, а не заменяет
его. Файл эталона не коммитится. Скрипт проверяет принадлежность аудио и
гипотезы сохранённому прогону и сохраняет в отчёт только суммы и агрегаты.
При запуске без эталона WER остаётся null. `data_quality.json` содержит
контрольные суммы доказательств, определения знаменателей и статус WER.
Оценка моделей и настройка по финальному test этой командой не выполняются.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    result = build(args.reference)
    directory = ROOT / 'reports/nlp'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'data_quality.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (directory / 'data_quality.md').write_text(render(result), encoding='utf-8')
    print(json.dumps({'source_total': result['source_total'], 'prepared_total': result['prepared_total'],
                      'wer_status': result['debug_audio']['wer_status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
