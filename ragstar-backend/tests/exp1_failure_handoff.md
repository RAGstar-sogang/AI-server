# Exp1 Retrieval Failing Cases Handoff

이 문서는 `tests/test_exp1_retrieval.py`의 현재 `xfail` 케이스를 다른 팀에서 빠르게 처리하실 수 있도록 정리한 handoff 문서입니다.

현재 기준 테스트 상태:

- `24 passed`
- `16 xfailed`

여기서 `xfailed`는 “허용된 실패”가 아니라, **현재 본체 결함을 추적하기 위한 의도적 감시 항목**입니다.

## 우선순위 기준

- **상**: 실험 결과를 조용히 오염시키거나, 데이터 무결성을 깨뜨리거나, 실패를 숨길 수 있는 경로입니다.
- **중**: 실험 harness가 비정상 입력/응답을 방어하지 못해 잘못된 평가를 만들 수 있는 경로입니다.
- **하**: 상대적으로 드문 입력 이상이지만, 계약 관점에서 명확히 막아야 하는 경로입니다.

---

## 상

| Case | 현재 실패 경로 커버 범위 | 왜 중요한가 | 테스트를 통과시키려면 필요한 변경 |
|---|---|---|---|
| `test_load_oom_logs_by_id_should_reject_duplicate_log_id` | `load_oom_logs_by_id()`가 중복 `log_id`를 만나도 마지막 row로 조용히 덮어씀 | 같은 로그가 다른 원문으로 치환되어 평가 결과가 조용히 왜곡될 수 있음 | `load_oom_logs_by_id()`에서 `log_id` 중복 검사 후 `ValueError` 발생 |
| `test_load_oom_logs_by_id_should_reject_row_missing_log_id` | `oom_logs.jsonl` row에 `log_id`가 없어도 dict comprehension 단계에서 비정상 동작 | join 키가 없으면 실험 데이터와 원문 로그 연결 자체가 깨짐 | 각 row에 `log_id` 존재 여부를 먼저 검사하고, 없으면 구조화된 예외 발생 |
| `test_load_oom_logs_by_id_should_reject_non_string_raw_log` | `raw_log`가 문자열이 아니어도 그대로 로딩됨 | raw query / parser 입력으로 부적절한 타입이 흘러가며 downstream 오작동 가능 | `raw_log` 타입을 `str`로 강제 검증하고 아니면 예외 발생 |
| `test_main_should_fail_when_log_id_join_is_missing_in_raw_mode` | `qa_ground_truth.jsonl`의 `log_id`가 `oom_logs.jsonl`에 없어도 빈 문자열 query로 계속 진행 | 데이터 join 실패가 조용히 recall 0으로 변장할 수 있음 | `main()` 또는 join helper에서 `log_id` 미매칭 시 즉시 `ValueError` 발생 |
| `test_main_should_fail_when_raw_mode_query_string_is_empty` | raw 모드에서 빈 `raw_log`를 그대로 `collection.query()`에 전달 | 빈 query로 검색이 수행되면 결과가 무의미하고 디버깅도 어려움 | raw 모드 query 직전 빈 문자열 검사 후 예외 발생 |
| `test_main_should_fail_when_raw_log_is_empty_in_parsed_mode` | parsed 모드에서도 빈 `raw_log`를 parser/query 경로로 흘림 | parser 결과와 검색 결과가 모두 신뢰 불가 | parsed 모드 진입 전에 `raw_log.strip()` 검증 추가 |
| `test_main_raw_mode_should_surface_query_error` | raw 모드 `collection.query()` 예외를 `except Exception: pass`로 삼킴 | 실험 실패가 recall 0처럼 보이는 가장 위험한 silent failure | raw 모드에서 예외를 재전파하거나, 최소한 구조화된 에러 상태로 수집 후 실행 실패 처리 |
| `test_main_raw_mode_should_reject_partially_corrupted_query_response` | raw 모드가 `ids` 길이 불일치 같은 손상 응답을 정상 응답처럼 취급 | retrieval 결과가 일부만 반영되어 평가가 왜곡됨 | raw query 결과의 `ids/documents/distances/metadatas` shape 정합성 검사 추가 |

---

## 중

| Case | 현재 실패 경로 커버 범위 | 왜 중요한가 | 테스트를 통과시키려면 필요한 변경 |
|---|---|---|---|
| `test_main_should_reject_non_mapping_parsed_fields` | parser가 `parsed_fields`에 list 등 non-mapping을 반환해도 그대로 query builder로 전달 | parsed 모드 query formulation이 비정상 타입에 취약 | `build_parsed_retrieval_inputs()` 또는 `main()`에서 `parsed_fields`가 `dict`인지 검증 |
| `test_run_parsed_retrieval_query_should_handle_malformed_collection_response` | parsed retrieval이 손상된 응답 shape를 만나면 구조화된 에러로 정리하지 못함 | parsed 모드 retrieval이 런타임 오류나 잘못된 chunk 매핑으로 이어질 수 있음 | `run_parsed_retrieval_query()`에서 결과 shape 검사 후 에러 dict 반환 |
| `test_run_parsed_retrieval_query_should_handle_partially_corrupted_query_response` | `documents`/`ids`/`distances`/`metadatas` 길이 불일치를 가정하지 않음 | 일부 결과만 있는 손상 응답에서 chunk mapping이 깨질 수 있음 | 배열 길이 일치 여부 검증, 불일치 시 에러 반환 |
| `test_main_should_reject_ground_truth_row_with_non_list_relevant_chunk_ids` | `relevant_chunk_ids`가 list가 아니어도 사용됨 | recall 계산의 정답셋 정의가 무너짐 | dataset row validation 단계에서 `relevant_chunk_ids` 타입 검사 |
| `test_main_should_reject_ground_truth_row_with_non_string_expected_oom_type` | `expected_oom_type`이 dict 등 문자열이 아니어도 query/filter 구성에 사용됨 | parsed 모드 query / filter semantics가 깨짐 | dataset row validation에서 `expected_oom_type`을 `str`로 강제 |
| `test_main_should_reject_ground_truth_row_missing_log_id` | ground-truth row에 `log_id`가 없어도 fallback (`log_{i}`)로 흘러감 | 명시적 데이터 결함이 가려지고, 잘못된 join 실패로 바뀜 | `main()` dataset loop 초반에 `log_id` 필수성 검사 |

---

## 하

| Case | 현재 실패 경로 커버 범위 | 왜 중요한가 | 테스트를 통과시키려면 필요한 변경 |
|---|---|---|---|
| `test_main_should_reject_ground_truth_row_with_non_string_relevant_chunk_id_items` | `relevant_chunk_ids` 내부 원소가 숫자/객체여도 허용됨 | recall label 품질은 깨지지만 주로 데이터 정제 문제에 가까움 | list 내부 원소가 모두 문자열인지 검증 |
| `test_main_should_reject_empty_ground_truth_row` | `{}` 같은 빈 row를 fallback 값으로 처리함 | 데이터 오류를 “합법적인 unknown case”처럼 숨김 | empty row 검사 후 즉시 예외 발생 |

---

## 구현 권장 순서

1. **입력 검증 유틸을 추가해 주십시오**
	- `qa_ground_truth.jsonl` row validator
	- `oom_logs.jsonl` row validator
	- 빈 문자열 `raw_log` validator

2. **join / query 실행 전에 조기 실패(fail fast)를 적용해 주십시오**
	- `log_id` join 실패 시 즉시 중단
	- raw / parsed 모두 empty log 차단
	- raw 모드 예외 삼키기 제거

3. **query 결과 shape validator를 추가해 주십시오**
	- raw / parsed 공통으로 `documents`, `ids`, `distances`, `metadatas` 정합성 검사
	- 손상 응답은 빈 결과가 아니라 구조화된 에러 또는 명시적 예외로 처리

4. **테스트 해제 순서를 다음과 같이 권장드립니다**
	- 상 → 중 → 하 순으로 수정
	- 수정 후 각 `xfail`을 일반 passing test로 전환하거나 삭제 검토

---

## 빠른 참고: 케이스 목록

### 상
- `test_load_oom_logs_by_id_should_reject_duplicate_log_id`
- `test_load_oom_logs_by_id_should_reject_row_missing_log_id`
- `test_load_oom_logs_by_id_should_reject_non_string_raw_log`
- `test_main_should_fail_when_log_id_join_is_missing_in_raw_mode`
- `test_main_should_fail_when_raw_mode_query_string_is_empty`
- `test_main_should_fail_when_raw_log_is_empty_in_parsed_mode`
- `test_main_raw_mode_should_surface_query_error`
- `test_main_raw_mode_should_reject_partially_corrupted_query_response`

### 중
- `test_main_should_reject_non_mapping_parsed_fields`
- `test_run_parsed_retrieval_query_should_handle_malformed_collection_response`
- `test_run_parsed_retrieval_query_should_handle_partially_corrupted_query_response`
- `test_main_should_reject_ground_truth_row_with_non_list_relevant_chunk_ids`
- `test_main_should_reject_ground_truth_row_with_non_string_expected_oom_type`
- `test_main_should_reject_ground_truth_row_missing_log_id`

### 하
- `test_main_should_reject_ground_truth_row_with_non_string_relevant_chunk_id_items`
- `test_main_should_reject_empty_ground_truth_row`

