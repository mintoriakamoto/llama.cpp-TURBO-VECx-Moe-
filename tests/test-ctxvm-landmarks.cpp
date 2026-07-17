// CTX-VM landmark page-table reference test (docs/svmi-research.md §8, §11)
//
// Validates the two approximations the paged-context design rests on, in plain C++
// so CI guards them before any engine implementation lands:
//
//   1. mean-key landmarks rank pages nearly as well as exact per-key scoring
//      (recall@k of the landmark ranking vs the exact max-dot ranking)
//   2. product-quantized landmarks (§11, pq16) track the f16 landmark ranking
//      closely enough that the index can shrink 16x
//
// Synthetic keys are clustered with topical drift - the structure prose/code
// contexts exhibit; thresholds are set conservatively below the measured values
// (see scripts/svmi-pqindex.py for the tunable experiment).

// NOTE: std::normal_distribution is implementation-defined; this test must produce
// identical numbers on libstdc++/libc++/MSVC runners, so it uses its own Box-Muller
// over raw mt19937 output (which the standard does pin down).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static constexpr int N_PAGES      = 256;
static constexpr int TOKENS_PAGE  = 16;   // keys per page (small: keeps the test fast)
static constexpr int DIM          = 64;
static constexpr int N_QUERIES    = 64;
static constexpr int TOP_K        = 8;
static constexpr int PQ_M         = 16;   // sub-vectors
static constexpr int PQ_KCENT     = 32;   // centroids per sub-space
static constexpr int PQ_ITERS     = 6;

using vec = std::vector<float>;

static void normalize(vec & v) {
    float n = 0.0f;
    for (float x : v) n += x * x;
    n = std::sqrt(n) + 1e-12f;
    for (float & x : v) x /= n;
}

static float dot(const vec & a, const vec & b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static float recall_at(const std::vector<int> & ref, const std::vector<int> & got, int k) {
    int hit = 0;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (ref[i] == got[j]) { ++hit; break; }
        }
    }
    return (float) hit / (float) k;
}

static std::vector<int> rank_desc(const std::vector<float> & scores) {
    std::vector<int> idx(scores.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    return idx;
}

static float unif01(std::mt19937 & rng) {         // (0,1), bit-portable
    return ((rng() >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

static float gaussf(std::mt19937 & rng) {          // Box-Muller, bit-portable
    float u1 = unif01(rng), u2 = unif01(rng);
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}

int main() {
    std::mt19937 rng(42);

    // clustered keys with drift: consecutive pages share cluster centers
    const int n_clusters = N_PAGES / 16;
    std::vector<vec> centers(n_clusters, vec(DIM));
    for (auto & c : centers) { for (float & x : c) x = 2.0f * gaussf(rng); }

    std::vector<std::vector<vec>> pages(N_PAGES);
    std::vector<vec> landmarks(N_PAGES, vec(DIM, 0.0f));
    for (int p = 0; p < N_PAGES; ++p) {
        const vec & c = centers[p * n_clusters / N_PAGES];    // sorted assignment = drift
        pages[p].assign(TOKENS_PAGE, vec(DIM));
        for (auto & k : pages[p]) {
            for (int d = 0; d < DIM; ++d) k[d] = c[d] + 0.6f * gaussf(rng);
            normalize(k);
            for (int d = 0; d < DIM; ++d) landmarks[p][d] += k[d];
        }
        normalize(landmarks[p]);
    }

    // train PQ codebooks on the landmarks (k-means per sub-space)
    const int dsub = DIM / PQ_M;
    std::vector<std::vector<vec>> codebooks(PQ_M, std::vector<vec>(PQ_KCENT, vec(dsub)));
    std::vector<std::vector<uint8_t>> codes(N_PAGES, std::vector<uint8_t>(PQ_M));
    for (int m = 0; m < PQ_M; ++m) {
        for (int c = 0; c < PQ_KCENT; ++c) {
            const vec & src = landmarks[(size_t) c * N_PAGES / PQ_KCENT];
            codebooks[m][c].assign(src.begin() + m * dsub, src.begin() + (m + 1) * dsub);
        }
        for (int it = 0; it < PQ_ITERS; ++it) {
            std::vector<vec>  sums(PQ_KCENT, vec(dsub, 0.0f));
            std::vector<int>  cnt(PQ_KCENT, 0);
            for (int p = 0; p < N_PAGES; ++p) {
                int   best = 0;
                float bd   = 1e30f;
                for (int c = 0; c < PQ_KCENT; ++c) {
                    float d2 = 0.0f;
                    for (int d = 0; d < dsub; ++d) {
                        float df = landmarks[p][m * dsub + d] - codebooks[m][c][d];
                        d2 += df * df;
                    }
                    if (d2 < bd) { bd = d2; best = c; }
                }
                codes[p][m] = (uint8_t) best;
                ++cnt[best];
                for (int d = 0; d < dsub; ++d) sums[best][d] += landmarks[p][m * dsub + d];
            }
            for (int c = 0; c < PQ_KCENT; ++c) {
                if (cnt[c] > 0) {
                    for (int d = 0; d < dsub; ++d) codebooks[m][c][d] = sums[c][d] / (float) cnt[c];
                }
            }
        }
    }


    float rec_lm = 0.0f, rec_pq = 0.0f, rec_pq2 = 0.0f;
    for (int q = 0; q < N_QUERIES; ++q) {
        // queries look up existing content: perturb a real key
        vec query = pages[rng() % N_PAGES][rng() % TOKENS_PAGE];
        for (float & x : query) x += 0.4f * gaussf(rng);
        normalize(query);

        std::vector<float> s_exact(N_PAGES), s_lm(N_PAGES), s_pq(N_PAGES);
        for (int p = 0; p < N_PAGES; ++p) {
            float mx = -1e30f;
            for (const auto & k : pages[p]) mx = std::max(mx, dot(query, k));
            s_exact[p] = mx;
            s_lm[p]    = dot(query, landmarks[p]);
            float s = 0.0f;   // asymmetric PQ score via per-sub lookup
            for (int m = 0; m < PQ_M; ++m) {
                const vec & cent = codebooks[m][codes[p][m]];
                for (int d = 0; d < dsub; ++d) s += query[m * dsub + d] * cent[d];
            }
            s_pq[p] = s;
        }
        auto r_exact = rank_desc(s_exact);
        auto r_lm    = rank_desc(s_lm);
        auto r_pq    = rank_desc(s_pq);
        rec_lm  += recall_at(r_exact, r_lm, TOP_K);
        rec_pq  += recall_at(r_lm,    r_pq, TOP_K);
        // over-fetch recovery: PQ top-2k must contain most of the f16 top-k
        int hit = 0;
        for (int i = 0; i < TOP_K; ++i) {
            for (int j = 0; j < 2 * TOP_K; ++j) {
                if (r_lm[i] == r_pq[j]) { ++hit; break; }
            }
        }
        rec_pq2 += (float) hit / (float) TOP_K;
    }
    rec_lm  /= N_QUERIES;
    rec_pq  /= N_QUERIES;
    rec_pq2 /= N_QUERIES;

    printf("landmark  recall@%d vs exact scoring : %.3f (threshold 0.55)\n", TOP_K, rec_lm);
    printf("pq16      recall@%d vs f16 landmarks : %.3f (threshold 0.55)\n", TOP_K, rec_pq);
    printf("pq16 x2   f16 top-%d in pq top-%d    : %.3f (threshold 0.70)\n", TOP_K, 2 * TOP_K, rec_pq2);

    bool ok = rec_lm >= 0.55f && rec_pq >= 0.55f && rec_pq2 >= 0.70f;
    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
