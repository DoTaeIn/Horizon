# Horizon

게임 월드 데이터의 무손실 압축·복원을 연구하는 Python 프로토타입입니다. 현재는 uint16 높이맵(2D `.npy`)에 범위를 좁혀 코덱 구현, 복원 최적화, 실제 지형 및 디스크 읽기 포함 비교까지 진행했습니다. 장기 계획은 [PLAN.md](PLAN.md)에 있습니다.

## 현재까지의 결과

- 직접 Zstd, LZ4, 행 차분+Zstd의 압축·복원과 CLI를 구현했습니다.
- v1 컨테이너의 shape·값 바이트를 정확히 보존하며, CRC-32와 SHA-256으로 손상을 검사합니다.
- 큰 맵의 복원은 제한된 행 버퍼로 처리하고, 체크섬용 전체 배열 복사를 제거했습니다. 기존 압축 파일 바이트와 호환성을 유지했습니다.
- 공개 Skadi 한국·알프스 지형 각각 3601×3601과 중앙 크롭으로 정확 복원 및 성능을 검증했습니다.
- 실제 파일 읽기부터 복원 완료까지, 파일 캐시 사용과 Windows OS 파일 캐시 우회를 비교했습니다.
- 최신 테스트 실행: **130개 중 127개 통과, 3개 건너뛰기**. Windows 네이티브 캐시 우회 검사는 실제 실행했고, 건너뛴 항목은 심볼릭 링크 권한 검사 2개와 비-Windows 전용 검사 1개입니다.

개발은 Claude CLI의 Opus·Sonnet 병렬 작업과 별도 감사·실측으로 진행했습니다. 큰 맵 성능 퇴행, 캐시 NPY 헤더 검증 누락, 읽기 버퍼의 과도한 할당을 찾아 수정했으며 반려한 결과는 최종 측정과 구분했습니다.

### 복원 최적화

같은 실행에서 이전 코덱과 현재 코덱을 교대로 50회 측정했습니다. 전체 지형의 차분+Zstd 결과입니다.

| 지형 | 이전 중앙값 | 현재 중앙값 | 시간 감소 | 추적 가능한 피크 메모리 |
| --- | ---: | ---: | ---: | ---: |
| 한국 N37E127 | 98.82ms | 85.55ms | 13.44% | 85.80 → 60.39MB |
| 알프스 N46E007 | 100.24ms | 86.46ms | 13.74% | 86.55 → 61.13MB |

작은 합성 맵은 사실상 이전 수준입니다. 메모리는 `tracemalloc` 측정이며 프로세스 RSS와 추적되지 않는 네이티브 할당은 제외합니다. [원시 결과](benchmarks/decode-real-v2.json) · [감사](benchmarks/decode-audit.md)

### 디스크 읽기+복원

현재 직접 Zstd와 차분+Zstd를 조건·방식별 50회씩, 총 400회 측정한 중앙값입니다. 각 호출은 파일 열기·읽기·닫기와 복원·내부 체크섬을 포함합니다.

| 전체 지형·조건 | 직접 Zstd | 차분+Zstd |
| --- | ---: | ---: |
| 한국·캐시 사용 | 82.84ms | 86.58ms |
| 알프스·캐시 사용 | 84.60ms | 86.96ms |
| 한국·OS 캐시 우회 | 92.72ms | 92.88ms |
| 알프스·OS 캐시 우회 | 93.54ms | 94.71ms |

**차분 방식의 확실한 전체 로딩 속도 우위는 확인하지 못했습니다.** 캐시 사용 시 직접 Zstd가 빨랐고, 캐시 우회 시에는 거의 동률이었습니다. 차분 컨테이너는 직접 Zstd보다 한국 33.50%, 알프스 40.77% 작았습니다. 현재 확인된 이점은 저장 공간 절감입니다.

측정 환경은 단일 Windows 장치, Python 3.10.6, NumPy 2.2.6, zstandard 0.25.0, lz4 4.4.5입니다. OS 캐시 우회는 SSD·컨트롤러 캐시나 메타데이터까지 비운 완전한 콜드 로딩이 아닙니다. p95·원시 표본·환경·파일 해시는 [디스크 결과](benchmarks/disk-load-real-v2.json), 방법과 반려한 수정은 [감사 보고서](benchmarks/disk-load-audit.md)에 있습니다.

### 남은 검증

실제 수제 게임 높이맵, 다른 저장장치·하드웨어, 완전한 콜드 부팅, 엔진·GPU·충돌 준비까지의 로딩, 사전 Zstd, 규칙 추론과 월드 생성은 아직 검증하거나 구현하지 않았습니다. 현재 결과를 게임 전체 로딩 성능으로 일반화하지 않습니다.

## 파일 구성

- `horizon/benchmark.py`: `run_benchmark(size=256, repeats=30, inputs=())`
- `horizon/__main__.py`: `python -m horizon` CLI (argparse)
- `horizon/codec.py`: `encode(heightmap, method="delta_zstd")`, `decode(blob)`
- `tests/`: 코덱 정확도, 손상 입력, 벤치마크, CLI 테스트
- `benchmarks/`: 합성·실제 지형 측정 결과와 감사 보고서
- `scripts/`: 공개 DEM 준비와 변경 전후 복원 속도 비교

## 설치와 테스트

```
uv sync
uv run python -m unittest discover -s tests -v
```

Python 3.10 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다. 의존성 잠금은 `uv.lock`에 포함됩니다. 일반 코덱은 Python/NumPy로 동작하며, 디스크 캐시 우회 벤치마크는 Windows 전용입니다.

공개 저장소에는 코드·테스트·측정 JSON·감사 문서·작은 지형 출처 메타데이터를 포함합니다. 대용량 지형 원본, 생성 컨테이너, 가상환경, Claude 실행 로그는 제외합니다.

실제 데이터는 새 디렉터리로 받습니다. 저장소에 있는 `data/terrain/*.json`은 당시 측정의 출처·해시 기록이며, 해당 경로에는 원본 파일을 배포하지 않습니다.

```powershell
uv run python scripts/prepare_terrain.py --output-dir data/terrain-local
uv run python scripts/compare_decode.py --input data/terrain-local/N37E127.npy --input data/terrain-local/N46E007.npy --repeats 50 --output benchmarks/my-decode.json
```

데이터 출처·저작자 표시·무손실 정규화·재현 방법은 [지형 문서](data/terrain/README.md)를 참고하세요. 측정 JSON과 컨테이너 디렉터리는 기존 경로를 덮어쓰지 않으므로 재실행할 때 새 이름을 사용합니다.

## 사용법

벤치마크 (출력 파일의 상위 폴더가 미리 존재해야 합니다):

```
uv run python -m horizon benchmark --output benchmarks/example.json
uv run python -m horizon benchmark --size 256 --repeats 30
uv run python -m horizon benchmark --input real_map.npy --input another.npy
```

`--output`을 생략하면 JSON 리포트를 표준 출력으로 내보냅니다.

압축/복원:

```
uv run python -m horizon pack INPUT.npy OUTPUT.hzn --method delta_zstd
uv run python -m horizon unpack INPUT.hzn OUTPUT.npy
```

사용 가능한 방식: `zstd`, `lz4`, `delta_zstd`(기본값).

## 알아둘 점

- 입력은 비어 있지 않은 2D little-endian **uint16** 배열만 허용합니다. 암묵적 dtype 변환은 없습니다. 원본 크기 상한은 64 MiB입니다.
- 값 바이트와 shape는 그대로 보존되지만 `.npy` 헤더 바이트까지 동일함을 보장하지는 않습니다.
- 컨테이너 포맷 버전은 고정되어 있으며, 손상/잘림/뒤따르는 데이터는 `ValueError`로 거부됩니다.
- 포맷 v1은 60바이트 헤더에 크기·코덱·예측기·원본 SHA-256·헤더 CRC-32를 저장합니다. 원본과 잔차 Zstd는 모두 level 3이며, LZ4는 프레임 기본 설정을 사용합니다.
- `delta_zstd`는 행 첫 값을 유지하고 나머지 값을 왼쪽 이웃과의 차분(mod 65536)으로 저장한 뒤 Zstd로 압축합니다. 자동 규칙 추론은 후속 단계입니다.
- 값 배열을 할당하기 전에 NPY v1/v2/v3 헤더의 dtype·shape·실제 데이터 길이를 검증합니다. 객체 배열, 잘린 파일, 후행 데이터는 거부합니다.
- 기존 출력 파일은 절대 덮어쓰지 않습니다. 대상이 이미 있으면 오류 메시지와 함께 0이 아닌 종료 코드로 끝나며, 검증과 인코딩/디코딩을 마친 뒤에만 출력 파일을 만듭니다.
- 모든 데이터셋(flat, plane, smooth terrain, 시드 고정 random)은 `--input`으로 실제 파일을 주지 않는 한 합성 데이터입니다. 고정 시드는 리포트에 기록됩니다.
- 측정값은 맵 전체 컨테이너의 압축/해제 시간이며, 스트리밍이나 엔진 성능을 의미하지 않습니다.
- 사전(dictionary) Zstd는 이후 비교할 기준선(baseline)으로 남겨 두었고, 생성(generation) 기능은 구현되어 있지 않습니다.

## 리포트 형식

`schema_version`, `environment`(Python, 플랫폼, NumPy/zstandard/lz4 버전), `settings`(size, repeats, warmup, seed, methods), `results`로 구성됩니다. `results`의 각 항목은 맵/방식별로 `sha256`, `shape`, `raw_bytes`, `container_bytes`, `ratio`(container/raw), `encode_median_ms`, `decode_median_ms`, `decode_p95_ms`, `exact_match`를 가집니다. 모든 반복의 디코드 결과는 시간 측정 구간 밖에서 바이트와 shape가 정확히 일치하는지 검사합니다.

워밍업은 방식별 1회, 기본 측정은 30회입니다. p95는 NumPy의 선형 보간 백분위수로 계산합니다. 코덱 호출 시간에는 체크섬 및 컨테이너 처리도 포함됩니다. `settings`에는 압축 설정과 컨테이너 버전·헤더 크기를 함께 기록합니다. 원본 64 MiB는 입력 데이터 제한이며, 실제 작업 메모리는 복사본과 합성 데이터 생성 때문에 그보다 큽니다.

초기 측정 결과와 후속 개선은 [benchmarks/README.md](benchmarks/README.md)에 기록했습니다. 공개 한국·알프스 DEM으로 이전 코덱과 현재 코덱을 비교했고, 정확한 복원과 v1 컨테이너 호환성을 확인했습니다. 데이터 준비와 재현 명령은 [data/terrain/README.md](data/terrain/README.md), 감사 결과는 [benchmarks/decode-audit.md](benchmarks/decode-audit.md)에 있습니다. 실제 게임 월드와 엔진 성능은 아직 확인하지 않았습니다.

## 디스크 읽기 포함 비교 (Windows)

현재 직접 Zstd와 차분+Zstd를 파일 열기·읽기·닫기부터 복원 완료까지 비교합니다.
`warm_buffered`는 매번 측정 직전에 파일 전체를 읽어 캐시를 준비하고,
`os_cache_bypass`는 Windows `FILE_FLAG_NO_BUFFERING`으로 OS 파일 데이터 캐시를
우회합니다. SSD·컨트롤러 캐시나 파일 메타데이터까지 비운 콜드 부팅 측정은 아닙니다.

```powershell
uv run python scripts/compare_disk.py --input data/terrain-local/N37E127.npy --input data/terrain-local/N46E007.npy --containers-dir data/disk-load-new --repeats 50 --output benchmarks/disk-load-new.json
```

출력 JSON과 컨테이너 디렉터리는 모두 새 경로여야 합니다. 큰 컨테이너 파일은
`data/disk-load-*/`에서 Git 제외됩니다. Windows 캐시 우회가 실패하면 전체 측정이
실패하며 일반 읽기로 대체하지 않습니다. 출력 디렉터리와 원본 데이터 준비 방법은
위 지형 문서를 참고하세요. 새 체크아웃에서 데이터 경로를 바꾸면 `--input`도 바꿉니다.

두 전체 타일을 조건·방식별 50회 측정했습니다. 캐시 사용 시 직접 Zstd가 더 빨랐고,
캐시 우회 시 차분+Zstd의 명확한 전체 로딩 우위는 확인하지 못했습니다.
[측정 결과](benchmarks/disk-load-real-v2.json)와 [감사 보고서](benchmarks/disk-load-audit.md)에
전체·읽기·복원 중앙값 및 p95, 원시 표본, 파일 해시, 캐시 조건을 기록했습니다.
