# 특이값 분해(SVD): 선형변환을 방향과 크기로 나누기

> 바탕 영상: 혁펜하임, [「[선대] 6-1강. 특이값 분해 (SVD: Singular Value Decomposition) 의 모든 것!」](https://www.youtube.com/watch?v=TxB96QVlgXk) (15:34, 2022-10-04).
> 2026-09-20에 Computer use의 YouTube 스크립트 내보내기로 한국어 자동 생성 자막을 가져와 확인했다. 이 문서는 전문 전사가 아니라 수식과 용어를 검토해 재구성한 설명문이다. 1–6절은 영상의 주제 순서를 따르며, 7–9절의 수치 예제와 연구 연결은 이해를 위한 보충이다.

## 1. 고유값 분해에서 SVD로

행렬 $A$를 벡터를 바꾸는 함수로 생각하자. $y=Ax$는 입력 $x$를 출력 $y$로 보내는 **선형변환**이다. SVD는 이 변환을 “입력 방향을 잡고, 방향별로 크기를 바꾸고, 출력 방향으로 조립하는 과정”으로 나눈다.

영상은 [0:00](https://www.youtube.com/watch?v=TxB96QVlgXk&t=0s)부터 고유값 분해의 한계와 비교한다. 이때 두 종류의 분해를 구별해야 한다.

- 일반적인 고유값 분해 $A=P\Lambda P^{-1}$: 정사각행렬이면서 충분한 수의 선형 독립 고유벡터를 가져야 한다. 대칭일 필요는 없다.
- 실수 대칭행렬의 직교 고유값 분해 $A=Q\Lambda Q^\top$: 고유벡터들을 정규직교 기저로 선택할 수 있다.
- SVD $A=U\Sigma V^\top$: **모든 실수 $m\times n$ 행렬에 존재**한다. 직사각행렬이나 랭크가 부족한 행렬에도 적용된다.

따라서 영상의 “대칭이어야 한다”는 설명은 일반적인 대각화의 필요조건으로 이해하면 안 된다. 실수 행렬을 직교행렬과 실수 대각행렬로 $Q\Lambda Q^\top$처럼 분해하는 맥락이다.

## 2. 세 행렬이 뜻하는 것

$A\in\mathbb R^{m\times n}$의 완전 SVD(full SVD)는 다음과 같다.

$$
A=U\Sigma V^\top,
\qquad
U\in\mathbb R^{m\times m},\quad
\Sigma\in\mathbb R^{m\times n},\quad
V\in\mathbb R^{n\times n}.
$$

| 기호 | 구성 | 의미 |
|---|---|---|
| $V=[v_1\;\cdots\;v_n]$ | 오른쪽 특이벡터 | 입력 공간의 정규직교 방향들 |
| $\Sigma$ | 대각 위치의 특이값, 나머지는 0 | 방향별 늘이기·줄이기 및 차원 사이의 대응 |
| $U=[u_1\;\cdots\;u_m]$ | 왼쪽 특이벡터 | 출력 공간의 정규직교 방향들 |

$U^\top U=UU^\top=I_m$, $V^\top V=VV^\top=I_n$이다. **정규직교**란 각 벡터의 길이가 1이고 서로 다른 벡터의 내적은 0이라는 뜻이다.

$p=\min(m,n)$이라 하면 특이값은 보통

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_p\ge0
$$

순서로 둔다. $\Sigma$는 직사각형일 수 있으므로 정사각 대각행렬과 구분해 생각한다. 특이값은 음수가 아니며, 0도 가능하다. 이 정의와 차원은 [MIT SVD 강의 슬라이드](https://math.mit.edu/~gs/linearalgebra/ila6/SVD_Slides.pdf)에서도 확인할 수 있다.

## 3. 왜 $A^\top A$와 $AA^\top$를 사용하는가

영상 [3:34](https://www.youtube.com/watch?v=TxB96QVlgXk&t=214s)의 핵심은 SVD와 고유값 분해를 연결하는 것이다. $A$가 어떤 모양이든 $A^\top A$와 $AA^\top$는 정사각 대칭행렬이다. 또한

$$
x^\top A^\top Ax=\|Ax\|^2\ge0
$$

이므로 $A^\top A$는 양의 준정부호이고, 고유값이 모두 0 이상이다. $AA^\top$도 같은 성질을 갖는다.

SVD를 대입하면 중간의 직교행렬들이 소거된다.

$$
A^\top A=V\Sigma^\top\Sigma V^\top,
\qquad
AA^\top=U\Sigma\Sigma^\top U^\top.
$$

따라서 $v_i$는 $A^\top A$의 고유벡터이고 $u_i$는 $AA^\top$의 고유벡터다. 대응하는 고유값은 $\sigma_i^2$다. 즉 **특이값은 $A$ 자체의 고유값이 아니라, $A^\top A$의 고유값의 음이 아닌 제곱근**이다. 두 곱행렬은 양의 고유값과 그 중복도가 같지만, 크기가 다르면 0인 고유값의 개수는 다를 수 있다.

손으로 분해를 구할 때는 다음처럼 연결하면 부호와 순서의 불일치를 피할 수 있다.

1. $A^\top A$의 정규직교 고유벡터 $v_i$와 고유값 $\lambda_i$를 구한다.
2. 고유값이 큰 순서대로 정렬하고 $\sigma_i=\sqrt{\lambda_i}$로 둔다.
3. $\sigma_i>0$인 경우 $u_i=Av_i/\sigma_i$로 정한다.
4. 완전 SVD가 필요하면 남은 출력 방향들을 정규직교 기저로 보완한다.

3번의 벡터는 실제로 단위 벡터다.

$$
\|u_i\|^2
=\frac{v_i^\top A^\top Av_i}{\sigma_i^2}
=1.
$$

서로 다른 $i,j$에 대해서도 $u_i^\top u_j=0$이 성립한다. $\sigma_i=0$이면 나눌 수 없으므로 이 공식을 적용하지 않는다. 완전 SVD의 남은 $u_i$는 $\ker(A^\top)$의 정규직교 기저로 채운다.

같은 특이값이 반복되면 해당 부분공간 안의 기저는 유일하지 않다. $U$와 $V$를 각각 고유값 분해로 구했다면, 같은 특이값에 해당하는 벡터들을 $Av_i=\sigma_i u_i$가 성립하도록 맞춰야 한다.

## 4. 0이 아닌 특이값의 개수가 랭크다

영상 [8:04](https://www.youtube.com/watch?v=TxB96QVlgXk&t=484s)에서 강조하는 관계는

$$
\operatorname{rank}(A)=\#\{i:\sigma_i>0\}=r
$$

이다. $U$와 $V^\top$는 가역이므로 좌우에 곱해도 랭크를 바꾸지 않는다. 결국 $\operatorname{rank}(A)=\operatorname{rank}(\Sigma)$이고, $\Sigma$의 랭크는 0이 아닌 대각 성분의 개수다.

기하학적으로는 **입력의 독립적인 방향 중 출력에 살아남는 방향의 수**다. 0에 해당하는 입력 방향은 $Av_i=0$이 되어 사라진다. 실제 부동소수점 계산에서는 아주 작은 값도 오차로 나타날 수 있어, 수치 랭크는 허용오차를 정해 판정한다.

## 5. 첫 번째 기하학적 해석: 특별한 입력 방향과 출력 방향

영상 [9:25](https://www.youtube.com/watch?v=TxB96QVlgXk&t=565s)처럼 $A=U\Sigma V^\top$의 오른쪽에 $V$를 곱하면 $AV=U\Sigma$다. 양의 특이값에 대응하는 열을 하나씩 보면

$$
\boxed{Av_i=\sigma_i u_i}
$$

를 얻는다. 길이 1인 입력 $v_i$는 $A$를 통과하면서 방향이 $u_i$로 바뀌고 길이가 $\sigma_i$가 된다.

서로 직교하는 **특이벡터 방향들**은 변환 후에도 서로 직교한다. 이것이 모든 직교 입력 쌍에 대해 성립하는 것은 아니다. 일반적인 $A$는 각도를 바꿀 수 있다. 또 $\sigma_i=0$인 출력은 영벡터라서 방향 자체가 없다.

## 6. 두 번째 기하학적 해석: 좌표 변환 → 크기 조절 → 재조립

영상 [11:28](https://www.youtube.com/watch?v=TxB96QVlgXk&t=688s) 이후에는 임의의 입력 $x$를 생각한다. 행렬 곱은 오른쪽부터 작용한다.

```text
x ∈ ℝⁿ  ── Vᵀ ──▶  입력 특이벡터 기준의 좌표
        ── Σ  ──▶  좌표별 크기 조절, 영방향 제거, 출력 차원에 대응
        ── U  ──▶  출력 특이벡터들을 조합한 Ax ∈ ℝᵐ
```

이를 한 식으로 쓰면

$$
Ax=\sum_{i=1}^{r}\sigma_i u_i(v_i^\top x).
$$

$v_i^\top x$는 입력이 $v_i$ 방향을 얼마나 포함하는지 나타내는 좌표다. 그 좌표에 $\sigma_i$를 곱하고 $u_i$ 방향으로 출력한다. 모든 살아남은 방향의 출력을 더한 것이 $Ax$다.

직교행렬 $Q$는 $\|Qx\|^2=x^\top Q^\top Qx=\|x\|^2$이므로 길이를 보존하고 내적과 각도도 보존한다. 따라서 $U$와 $V^\top$는 회전이나 반사에 해당하고, 길이 변화는 $\Sigma$가 담당한다. 2차원에서는 단위원이 타원으로 바뀌는 모습으로 볼 수 있다. 타원의 축 방향은 $u_i$, 반축 길이는 $\sigma_i$다. 랭크가 1이면 타원이 선분으로 퇴화한다.

### 영상 설명에서 보완할 점

자동 자막의 [12:26 부근](https://www.youtube.com/watch?v=TxB96QVlgXk&t=746s)에는 직교행렬의 고유값이 항상 $\pm1$이라는 설명이 나온다. **실수 고유값이라면 맞지만 일반적으로는 복소수 고유값도 가능하며, 정확한 조건은 $|\lambda|=1$**이다.

예를 들어 $Q=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$는 90도 회전인 직교행렬이며 고유값은 $\pm i$다. 길이 보존은 고유값을 실수로 가정하지 않고 위의 $Q^\top Q=I$로 바로 증명하는 편이 정확하다. 또한 $\det Q=\pm1$이지만, 행렬식의 절댓값이 1이라는 조건만으로 길이 보존을 결론 내릴 수는 없다. $\operatorname{diag}(2,1/2)$는 넓이를 보존하면서 길이를 바꾼다.

## 7. 보충 예제: 직접 계산해 보기

영상의 예제를 전사한 것이 아니라, 계산을 확인하기 위한 별도 예제다.

$$
A=\begin{bmatrix}0&2\\3&0\end{bmatrix},
\qquad
A^\top A=\begin{bmatrix}9&0\\0&4\end{bmatrix}.
$$

고유값은 9와 4이므로 특이값은 3과 2다. 입력 특이벡터를 $v_1=(1,0)^\top$, $v_2=(0,1)^\top$로 두면

$$
u_1=\frac{Av_1}{3}=\begin{bmatrix}0\\1\end{bmatrix},
\qquad
u_2=\frac{Av_2}{2}=\begin{bmatrix}1\\0\end{bmatrix}.
$$

따라서

$$
U=\begin{bmatrix}0&1\\1&0\end{bmatrix},
\quad\Sigma=\begin{bmatrix}3&0\\0&2\end{bmatrix},
\quad V=I_2,
\qquad U\Sigma V^\top=A.
$$

$x=(1,1)^\top$는 먼저 좌표가 유지되고, $\Sigma$에 의해 $(3,2)^\top$, $U$에 의해 $(2,3)^\top$가 된다. 직접 $Ax$를 계산한 결과와 같다.

## 8. 보충: 저랭크 근사와 Low-Rank MoE의 연결

영상은 마지막에 데이터 압축과 PCA를 다음 응용으로 예고한다. 아래는 그 연결을 위한 추가 설명이다.

양의 특이값 $r$개만 남기는 compact SVD는 원래 행렬을 정확하게 표현한다.

$$
A=U_r\Sigma_r V_r^\top
=\sum_{i=1}^{r}\sigma_i u_i v_i^\top.
$$

각 항 $\sigma_i u_i v_i^\top$는 랭크 1 행렬이다. 이제 $k<r$개만 남기는 truncated SVD를 만들면

$$
A_k=U_k\Sigma_k V_k^\top
=\sum_{i=1}^{k}\sigma_i u_i v_i^\top
$$

가 된다. 이는 랭크가 최대 $k$인 행렬 중 Frobenius 노름의 근사 오차를 최소화한다. 오차와 보존되는 제곱 크기 비율은

$$
\|A-A_k\|_F^2=\sum_{i=k+1}^{r}\sigma_i^2,
\qquad
\frac{\|A_k\|_F^2}{\|A\|_F^2}
=\frac{\sum_{i=1}^{k}\sigma_i^2}{\sum_{i=1}^{r}\sigma_i^2}
\quad(A\ne0).
$$

원래 $mn$개 값 대신 $B=U_k\Sigma_k\in\mathbb R^{m\times k}$와 $C=V_k^\top\in\mathbb R^{k\times n}$를 저장하면 $k(m+n)$개 값으로 $A_k=BC$를 표현한다. $k(m+n)<mn$일 때 행렬 값의 저장량이 줄어든다.

MoE의 expert 가중치 $W$에도 이 표현을 생각할 수 있다. 다만 **가중치 자체의 Frobenius 오차가 최소라는 것과 모델 출력 오차가 최소라는 것은 다르다.** $\Delta=W-W_k$라 하면 입력에 대한 평균 제곱 출력 오차는

$$
\mathbb E\|\Delta x\|^2
=\operatorname{tr}\!\left(\Delta\,\mathbb E[xx^\top]\,\Delta^\top\right)
$$

처럼 입력 분포에 의존한다. 따라서 expert별 입력 분포와 라우팅을 고려한 성능 검증이 별도로 필요하다. 이는 이 영상에서 검증된 MoE 결과가 아니라, SVD를 이 프로젝트에 적용할 때의 수학적 구분이다.

## 9. 보충: PCA와 연결하기

[PCA 설명](pca.md)처럼 평균을 뺀 $N$개의 $d$차원 관측값을 **열로** 쌓아 $X\in\mathbb R^{d\times N}$로 두자. $X=U\Sigma V^\top$이면

$$
C=\frac1NXX^\top
=U\frac{\Sigma\Sigma^\top}{N}U^\top.
$$

따라서 이 데이터 배치에서는 $U$의 열이 PCA 주성분 방향이고, 각 방향의 분산은 $\sigma_i^2/N$이다. 상위 $k$개 축으로 얻은 좌표는 $U_k^\top X$다. 관측값을 행으로 쌓는 관례에서는 주성분 방향이 $V$에 들어간다. **PCA와 SVD를 연결할 때는 먼저 데이터가 행에 있는지 열에 있는지 확인해야 한다.**

## 다시 확인할 질문

1. $A$가 $5\times3$일 때 완전 SVD의 $U,\Sigma,V$는 각각 어떤 크기인가?
2. 특이값이 $5,2,0$이라면 랭크는 얼마이며, 0에 해당하는 입력 방향은 어떻게 되는가?
3. 왜 $v_i$와 $u_i$를 구한 뒤 $Av_i=\sigma_i u_i$를 확인해야 하는가?

## 출처와 보정 범위

- [원본 영상](https://www.youtube.com/watch?v=TxB96QVlgXk): 정의, 두 곱행렬을 통한 분해, 랭크, 두 기하학적 해석의 구성에 사용했다. 본문의 시간 링크는 영상 설명란의 챕터 및 가져온 자막을 기준으로 한다.
- [MIT, Singular value decomposition of matrices](https://math.mit.edu/~gs/linearalgebra/ila6/SVD_Slides.pdf): SVD 정의와 행렬 차원의 교차 확인에 사용했다.
- 자동 자막의 발음 인식 오류는 `eigendecomposition`→고유값 분해, `orthogonal`→직교, `singular value`→특이값, `singular vector`→특이벡터로 정리했다. 수치 예제, 상세 유도, PCA·저랭크 연결은 별도로 작성했으며, 원본 강의의 발언으로 간주하지 않는다.
