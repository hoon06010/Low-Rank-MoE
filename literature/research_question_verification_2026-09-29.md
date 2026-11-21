# 표적 선행연구 확인 메모 — 2026-09-29

범위: `plans/2026-09-29-research-question-verification-plan.md` 단계 A. 검색일은 2026-09-29이다. 검색어는 `LatentMoE compression ratio shared experts original dimension N K`, `Towards Greater Leverage shared expert ratio section 4.1.3`, `LOW RANK EXPERTS ENABLE SPECIALIZATION IN DENSE TRANSFORMERS OpenReview`, `StructMoE PMLR 2024`였고, 아래 원문 주소를 직접 열었다. 이 기록은 표적 검색이며 전수 검색이나 최초성 판정이 아니다.

| 원문·확인 버전 | 확인 위치와 관찰 | 아직 확인하지 않은 것 |
|---|---|---|
| [LatentMoE](https://arxiv.org/html/2601.18089v1), arXiv:2601.18089v1 (2026-01-26) | §3 식 (1)–(2), §4.1 그림 3–5, 표 2. 공통 `d→ℓ→d` projection 사이에 routed expert가 있고 shared expert는 원래 `d` 차원에서 계산한다. `α=d/ℓ`에 따라 routed 수 `N′=αN`을 키우고, accuracy variant는 `K′=αK`도 키운다. 압축비·expert 확대 ablation이 이미 있다. | 원문에서 독립 seed 수를 찾지 못했다. iso-FLOP/parameter 성능은 저자 보고이며 이 프로젝트와 같은 고정 N/K·shared 폭·독립 gate/up/down factor rank 그리드는 아니다. |
| [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70.pdf), ACL 2024 최종본 | §§3.1–3.2, §4.3 그림 3. 작은 expert 폭, always-active shared expert, 총/활성 예산을 고려한 shared-count 비교의 선행. 폭 축소는 행렬 rank 제한과 다르다. | 코드 commit 및 seed 반복을 재확인하지 않았다. |
| [Towards Greater Leverage](https://arxiv.org/html/2507.17702v4), arXiv:2507.17702v4 (2025-10-21) | §2.1 및 §4.1.3 식 (9), 그림 7. 저자가 `M`을 model size라고 부르지만 §2.1에서는 비임베딩 forward FLOPs/token으로 정의한다. 이 `M`, activation ratio, 활성 expert 총수를 고정하고 routed 수를 shared 수로 치환한다. 손실은 공유 비율에 대해 U자형이라고 보고한다. | 동일 **총 파라미터** 통제로 해석하지 않는다. 독립 seed 수를 확인하지 않았다. |
| [Comprehensive Scaling Law](https://arxiv.org/html/2509.23678v1), arXiv:2509.23678v1 (2025-09-28) | §3 식 (2)–(3), §4–5. `S=n_s/G`이며 G는 활성 expert 수다. 총/활성 파라미터와 shared 비율을 다루지만 native factor rank는 없다. | 여러 seed 및 실험 코드 일치는 확인하지 않았다. |
| [Slicing and Dicing](https://arxiv.org/html/2605.11689v1), arXiv:2605.11689v1 | §3.2 식 `g_gen+Σk_i g_i=1`, 그림 6 및 부록 그림; active FFN 예산에서 generalist 추가가 시험 구성에서 동등하거나 악화됨. §3/Appendix C.1은 일부 seed 분석도 제공한다. | 그 결과를 이 프로젝트의 fixed-total/native-rank 비교에 직접 옮길 수 없다. |
| [StructMoE 2024](https://proceedings.mlr.press/v262/sarwar24a.html), PMLR 262:182–193 | 출판 페이지와 초록 확인. 주 expert 내부 secondary routing 및 activation 전 low-rank LoRE 결합. 별도 출력 shared FFN이 아니다. | 공식 코드와 수치 세부 대조는 다시 하지 않았다. |
| [StructMoE 2026 PDF ID 7eda…](https://openreview.net/pdf/7eda692ef4f9549b74c2de9283097ae444af2fd6.pdf) 및 [ID QSoc…](https://openreview.net/pdf?id=QSoc7HGc6Q) | 두 주소 모두 오늘 OpenReview browser-verification으로 차단됐다. 기존 `evidence_native_rank.md`와 `evidence_struct_shared.md`에는 §§3.1–3.5·Appendix B.1 rank/LoRE-count sweep의 과거 직접 읽기 기록이 있다. | 오늘 원문·revision·2026 게재 상태·두 ID의 동일성은 미검증. 과거 기록을 오늘 확인으로 둔갑시키지 않는다. |
| [LightMoE](https://arxiv.org/html/2603.12645v1), v1 | §§3.2–3.3, §4.1. pretrained OLMoE expert를 공유 base와 개별 residual로 사후 대체한다. | native 초기화 실험의 증거가 아니다. |
| [MLoRE](https://arxiv.org/html/2403.17749v1), v1 | §3 및 Appendix B.2 표 9. 시각 다중과제에서 full 3×3 generic path가 시험한 저랭크 generic path보다 좋았다. | 언어 모델의 예산 일치 상호작용 실험이 아니다. |
| [SD-MoE](https://arxiv.org/html/2602.12556v1), arXiv:2602.12556v1 (2026-02-13) | §§2–4, Appendix F. 전문가 행렬 안의 공통 spectral subspace와 고유 성분을 분리한다. | 별도 always-active FFN 효과가 아니다. |
| [Spectron](https://arxiv.org/html/2602.12429v2), arXiv:2602.12429v2 (2026-07-15) | §§3–6, Appendix B.2/B.4. native 저랭크 사전학습의 update spectral-norm 안정성 문제가 rank 결과의 경쟁 설명이 된다. | sparse expert 업데이트에서 같은 안정성 효과가 성립하는지는 미확인. |

## 좁힌 질문과 반대 증거

구조의 최초 결합은 기여로 주장하지 않는다. **고정 N/K의 언어 MoE에서 별도 shared SwiGLU 폭과 routed SwiGLU의 native gate/up/down factor rank가 품질에 상호작용하는가; 총 파라미터를 맞춘 HS↔LL 재배분과 각 셀에 대응하는 LR↔DW 대조에서 차이가 독립 seed 및 별도 동일 학습 FLOPs 비교에도 남는가?** LR 2×2 전체가 동일 총 파라미터인 것은 아니다. HS와 LL 및 중간 배분점 M이 총량을 맞추고, LS와 HL은 이 총량보다 각각 낮고 높다. 각 LR 셀에 대응하는 DW 셀의 총량·활성 수는 맞추되, 서로 다른 LR 셀 전체의 활성 수나 FLOPs가 같다는 뜻은 아니다. 이 질문은 LatentMoE의 공통 latent projection과 N/K 확장, StructMoE 2026의 dense base 내부 LoRE, DeepSeekMoE 및 scaling 연구의 shared-count 이동과 구별된다.

가장 강한 반대 결과는 Slicing and Dicing의 active-FFN 예산에서 generalist가 동등하거나 악화된 관찰이다. LatentMoE의 N 확장 없는 4배 압축이 품질을 떨어뜨린 점도 압축만으로 유지된다는 가정에 반한다. Towards Greater Leverage의 U자형 shared 비율은 무조건 넓은 shared가 좋다는 예측을 배제한다. 따라서 `I_LR>0`만으로 실제 품질 개선이나 계산 효율을 말할 수 없다. `G_low`, LR/DW 상호작용 `J`, 동일 FLOPs 비교 및 실패 seed가 필요하다. 직접 중복을 찾지 못했다는 판단은 위 범위로 한정한다.

검색 종료 기준: 현재 질문을 선점할 수 있는 직접 이웃과 핵심 반대 결과를 표에 담았으므로 실행 명세 검토로 넘어간다. OpenReview 접근이 회복되거나 직접 중복 후보가 나타나면 그 버전·예산축부터 재확인한다.
