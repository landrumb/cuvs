# 8-way clustering Pareto comparison

Merge time includes clustering, exact cross-input top-4 neighbor construction within each cluster, graph append/sort/cap, and CAGRA optimization. Recall is query recall@12 for the merged index.

| Dataset | Family | Configuration | Merge time (s) | Recall@12 | Pareto |
|---|---|---|---:|---:|:---:|
| OpenAI-2M | Flat balanced k-means | 2 iterations | 12.515 | 0.944438 |  |
| OpenAI-2M | Flat balanced k-means | 5 iterations | 24.411 | 0.946829 |  |
| OpenAI-2M | Flat balanced k-means | 10 iterations | 47.337 | 0.947304 |  |
| OpenAI-2M | Flat balanced k-means | 20 iterations | 72.929 | 0.947796 |  |
| OpenAI-2M | Flat balanced k-means | 1 iteration | 123.337 | 0.942142 |  |
| OpenAI-2M | Lloyd k-means tree, k=2 | k=2, 5 iterations/level | 6.840 | 0.937367 |  |
| OpenAI-2M | Lloyd k-means tree, k=5 | k=5, 5 iterations/level | 5.173 | 0.937246 |  |
| OpenAI-2M | Pivot tree | 1 repeat | 1.428 | 0.920125 | yes |
| OpenAI-2M | Pivot tree | 2 repeats | 1.893 | 0.936113 | yes |
| OpenAI-2M | Pivot tree | 4 repeats | 2.785 | 0.947117 | yes |
| OpenAI-2M | Pivot tree | 8 repeats | 4.637 | 0.956042 | yes |
| OpenAI-2M | Pivot tree | 16 repeats | 8.222 | 0.963504 | yes |
| OpenAI-2M | Pivot tree | 32 repeats | 16.093 | 0.968575 | yes |
| OpenAI-2M | Rebuild | Rebuild | 12.914 | 0.968525 | yes |
| Wiki-1M | Flat balanced k-means | 2 iterations | 2.647 | 0.981733 |  |
| Wiki-1M | Flat balanced k-means | 5 iterations | 3.783 | 0.982325 |  |
| Wiki-1M | Flat balanced k-means | 10 iterations | 5.278 | 0.982533 |  |
| Wiki-1M | Flat balanced k-means | 20 iterations | 7.597 | 0.982817 |  |
| Wiki-1M | Flat balanced k-means | 1 iteration | 8.720 | 0.984408 |  |
| Wiki-1M | Lloyd k-means tree, k=2 | k=2, 5 iterations/level | 1.959 | 0.978517 |  |
| Wiki-1M | Lloyd k-means tree, k=5 | k=5, 5 iterations/level | 1.263 | 0.979733 |  |
| Wiki-1M | Pivot tree | 1 repeat | 0.290 | 0.976558 | yes |
| Wiki-1M | Pivot tree | 2 repeats | 0.408 | 0.984808 | yes |
| Wiki-1M | Pivot tree | 4 repeats | 0.624 | 0.989625 | yes |
| Wiki-1M | Pivot tree | 8 repeats | 1.067 | 0.992533 | yes |
| Wiki-1M | Pivot tree | 16 repeats | 1.947 | 0.994625 | yes |
| Wiki-1M | Pivot tree | 32 repeats | 3.695 | 0.995317 | yes |
| Wiki-1M | Rebuild | Rebuild | 3.633 | 0.992242 |  |
| YFCC-10M | Flat balanced k-means | 2 iterations | 17.550 | 0.970157 |  |
| YFCC-10M | Flat balanced k-means | 5 iterations | 44.219 | 0.971737 |  |
| YFCC-10M | Flat balanced k-means | 10 iterations | 98.042 | 0.972182 |  |
| YFCC-10M | Flat balanced k-means | 1 iteration | 152.705 | 0.973459 |  |
| YFCC-10M | Flat balanced k-means | 20 iterations | 173.405 | 0.972112 |  |
| YFCC-10M | Lloyd k-means tree, k=2 | k=2, 5 iterations/level | 9.035 | 0.965306 |  |
| YFCC-10M | Lloyd k-means tree, k=5 | k=5, 5 iterations/level | 4.909 | 0.965291 |  |
| YFCC-10M | Pivot tree | 1 repeat | 1.297 | 0.956141 | yes |
| YFCC-10M | Pivot tree | 2 repeats | 1.478 | 0.966503 | yes |
| YFCC-10M | Pivot tree | 4 repeats | 1.842 | 0.974011 | yes |
| YFCC-10M | Pivot tree | 8 repeats | 2.550 | 0.980908 | yes |
| YFCC-10M | Pivot tree | 16 repeats | 4.049 | 0.986357 | yes |
| YFCC-10M | Pivot tree | 32 repeats | 7.135 | 0.989630 | yes |
| YFCC-10M | Rebuild | Rebuild | 23.001 | 0.988518 |  |
