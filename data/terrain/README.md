# 지형 데이터 (data/terrain)

공개 DEM 타일을 손실 없이 `uint16` 높이맵으로 바꿔 코덱 디코드 성능을 실제 지형으로 비교하기 위한 디렉터리입니다. 큰 원본 파일(`.hgt.gz`, `.npy`)은 Git에서 제외하고, 작은 `.json` 메타데이터는 출처와 해시 기록으로 보존합니다.

## 출처와 성격

- 데이터: Mapzen/Tilezen **Skadi** 지형 타일 (AWS Open Data, HGT 형식, gzip 압축).
- Skadi는 여러 공개 표고 자료를 합성(composite)한 실세계 DEM이며, 원본 NASA SRTM만으로 구성되었다고 보장할 수 없습니다.
- 지상 진실(ground truth) 정확도를 주장하지 않습니다. 응답 헤더에 원천 영상(source imagery) 정보가 없으면 메타데이터에 식별 불가로 기록하고 추정하지 않습니다.
- 공식 등록처: https://registry.opendata.aws/terrain-tiles/
- 출처 문서: https://github.com/tilezen/joerd/blob/master/docs/data-sources.md
- 저작자 표시 문서: https://github.com/tilezen/joerd/blob/master/docs/attribution.md

저작자 표시:

- Mapzen / Tilezen
- Terrain data courtesy of the U.S. Geological Survey.
- Produced using Copernicus data and information funded by the European Union - EU-DEM layers.
- NOAA / ETOPO1

## 타일

| 타일 | 지역 | 범위 |
| --- | --- | --- |
| `N37E127` | 한국 | 위도 37-38N, 경도 127-128E |
| `N46E007` | 알프스 | 위도 46-47N, 경도 7-8E |

각 타일은 3601 x 3601 샘플(1 arc-second)이며 행은 북쪽에서 남쪽, 열은 서쪽에서 동쪽 순서입니다. 가져오는 URL은 아래 두 개로 고정되어 있습니다.

- https://elevation-tiles-prod.s3.amazonaws.com/skadi/N37/N37E127.hgt.gz
- https://elevation-tiles-prod.s3.amazonaws.com/skadi/N46/N46E007.hgt.gz

## 정규화 (signed offset 전단사)

원본 HGT는 big-endian 부호 있는 int16(미터)입니다. 변환은 다음과 같고 손실이 없습니다.

- 정방향: `u = (h.astype(int32) + 32768).astype('<u2')`
- 역방향: `h = (u.astype(int32) - 32768).astype('>i2')`, 바이트열이 원본 HGT와 정확히 같음을 변환 시마다 검증합니다.
- nodata(`-32768`)는 `u = 0`으로 그대로 보존됩니다. 보간, 평활화, 클리핑, 리사이즈, 유리한 영역 선택은 하지 않습니다.
- 유효 고도의 최소/최대는 nodata를 제외하고 미터 단위로 메타데이터에 기록합니다.

## 산출물과 출처 증명(provenance)

타일마다 세 파일이 생성됩니다.

- `<tile>.hgt.gz`: 내려받은 원본 그대로
- `<tile>.npy`: 전체 타일 정규화 결과 (`<u2`, 3601x3601)
- `<tile>.json`: 메타데이터

메타데이터에는 정확한 URL, 지리적 범위, 내려받은 UTC 시각, 원천 영상 헤더(있을 때), gzip / 원본 HGT / 정규화 값 바이트 / `.npy` 파일의 SHA256, shape, 정규화 식과 역식, nodata 개수, 유효 고도 최소/최대, 등록처와 저작자 표시 링크가 들어 있습니다.

안전 규칙:

- 기존 출력은 덮어쓰지 않습니다(`xb`로만 생성).
- 세 파일이 모두 있으면 메타데이터와 캐시 해시, 변환 일치를 검증한 뒤에만 재사용합니다.
- 일부만 있거나 손상된 출력은 명확한 오류로 중단하며 삭제하지 않습니다.
- 새 출력 생성이 실패하면 해당 실행이 만든 파일만 정리합니다.
- 네트워크를 쓸 수 없을 때 합성(synthetic) 데이터로 대체하지 않습니다. 실패로 종료합니다.

## 실행 방법

```
uv run python scripts/prepare_terrain.py
```

특정 타일만: `--tile N37E127` (반복 가능), 출력 위치: `--output-dir`.

디코드 비교 (기준 코덱은 `benchmarks/reference/codec_v1.py`):

```
uv run python scripts/compare_decode.py --input data/terrain/N37E127.npy --input data/terrain/N46E007.npy --repeats 50 --output benchmarks/decode-real-v2.json
```

비교 범위는 합성 256 픽스처, 각 입력의 중앙 256x256 / 1024x1024 크롭(지도가 충분히 클 때만), 전체 타일입니다. 이 측정은 메모리에 올려 둔 컨테이너의 warm 디코드이며 디스크나 엔진 스트리밍 성능이 아닙니다. 다른 작업이 끝난 뒤에 실행하세요.

## 상태

두 전체 타일과 중앙 크롭을 실제 측정했습니다. 결과는 [복원 비교](../../benchmarks/decode-real-v2.json), 검증과 한계는 [감사 보고서](../../benchmarks/decode-audit.md)에 기록합니다. 한국 타일의 유효 고도는 -63~1460m, 알프스는 398~4469m이며 두 타일 모두 nodata는 0개입니다.

메타데이터만 있고 큰 파일이 없는 새 체크아웃에서는 기존 기록을 보존하기 위해 `--output-dir data/terrain-local`처럼 빈 디렉터리로 다운로드하고 비교 명령의 입력 경로도 바꾸세요. 기존 기록과 내려받은 파일의 SHA-256을 비교하면 같은 데이터를 썼는지 확인할 수 있습니다.
