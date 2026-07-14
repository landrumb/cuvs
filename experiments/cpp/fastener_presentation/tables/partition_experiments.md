# Fastener partition experiment tables

Times are end-to-end benchmark measurements on the experiment host. Partition CAGRA build time is excluded from merge time, consistently across variants.

## Pivot partition retention at the default eight repeats

| Dataset | Tree | Partition time (s) | Exact 12-NN same-leaf rate |
|---|---:|---:|---:|
| OpenAI-2M | pivot-binary | 1.951 | 0.279297 |
| OpenAI-2M | pivot-ternary | 3.651 | 0.303365 |
| Wiki-1M | pivot-binary | 0.507 | 0.523458 |
| Wiki-1M | pivot-ternary | 0.874 | 0.543457 |
| YFCC-10M | pivot-binary | 0.848 | 0.323995 |
| YFCC-10M | pivot-ternary | 1.033 | 0.347493 |

## Flat balanced k-means with k = ceil(n / 256)

This is one direct, non-hierarchical cuVS balanced k-means fit with 20 iterations and squared L2 distance.

| Dataset | Rows | Clusters (k) | Iterations | Time (s) | Min size | Mean size | Max size | Exact 12-NN same-cluster rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenAI-2M | 2,321,096 | 9,067 | 20 | 76.533 | 64 | 255.994 | 1644 | 0.476054 |
| Wiki-1M | 1,000,000 | 3,907 | 20 | 6.381 | 61 | 255.951 | 15355 | 0.550395 |
| YFCC-10M | 10,000,000 | 39,063 | 20 | 171.602 | 64 | 255.997 | 1330 | 0.340861 |

## Native NN-descent all-neighbors quality

| Dataset | k | Build time (s) | Exact 12-NN recall |
|---|---:|---:|---:|
| OpenAI-2M | 12 | 21.465 | 0.990417 |
| OpenAI-2M | 32 | 21.443 | 0.997437 |
| OpenAI-2M | 64 | 27.831 | 0.998433 |
| Wiki-1M | 12 | 6.685 | 0.997437 |
| Wiki-1M | 32 | 6.454 | 0.997660 |
| Wiki-1M | 64 | 7.640 | 0.998820 |
| YFCC-10M | 12 | 47.598 | 0.995280 |
| YFCC-10M | 32 | 47.662 | 0.997498 |
| YFCC-10M | 64 | 49.577 | 0.997599 |

## Merged-index results

| Dataset | Fan-in | Method | Merge time (s) | Recall@12 |
|---|---:|---|---:|---:|
| OpenAI-2M | 2 | Fastener binary | 4.608 | 0.965192 |
| OpenAI-2M | 2 | Fastener ternary | 6.520 | 0.965079 |
| OpenAI-2M | 2 | Native NN-descent k32 | 21.472 | 0.972004 |
| OpenAI-2M | 2 | Rebuild | 12.916 | 0.967929 |
| OpenAI-2M | 8 | Fastener binary | 4.595 | 0.957196 |
| OpenAI-2M | 8 | Fastener ternary | 6.490 | 0.957746 |
| OpenAI-2M | 8 | Native NN-descent k32 | 22.694 | 0.972221 |
| OpenAI-2M | 8 | Rebuild | 12.914 | 0.968525 |
| OpenAI-2M | 128 | Fastener binary | 4.372 | 0.926517 |
| OpenAI-2M | 128 | Fastener ternary | 6.267 | 0.929779 |
| OpenAI-2M | 128 | Native NN-descent k32 | 22.462 | 0.964046 |
| OpenAI-2M | 128 | Rebuild | 13.026 | 0.967746 |
| Wiki-1M | 2 | Fastener binary | 1.066 | 0.994475 |
| Wiki-1M | 2 | Fastener ternary | 1.491 | 0.993983 |
| Wiki-1M | 2 | Native NN-descent k32 | 6.688 | 0.995167 |
| Wiki-1M | 2 | Rebuild | 3.660 | 0.992417 |
| Wiki-1M | 8 | Fastener binary | 1.066 | 0.992675 |
| Wiki-1M | 8 | Fastener ternary | 1.481 | 0.992667 |
| Wiki-1M | 8 | Native NN-descent k32 | 6.674 | 0.994833 |
| Wiki-1M | 8 | Rebuild | 3.633 | 0.992242 |
| Wiki-1M | 128 | Fastener binary | 1.030 | 0.987208 |
| Wiki-1M | 128 | Fastener ternary | 1.442 | 0.987575 |
| Wiki-1M | 128 | Native NN-descent k32 | 6.653 | 0.993367 |
| Wiki-1M | 128 | Rebuild | 3.730 | 0.992400 |
| YFCC-10M | 2 | Fastener binary | 2.547 | 0.988035 |
| YFCC-10M | 2 | Fastener ternary | 2.801 | 0.987858 |
| YFCC-10M | 2 | Native NN-descent k32 | 48.948 | 0.991306 |
| YFCC-10M | 2 | Rebuild | 23.196 | 0.988603 |
| YFCC-10M | 8 | Fastener binary | 2.550 | 0.981045 |
| YFCC-10M | 8 | Fastener ternary | 2.814 | 0.981371 |
| YFCC-10M | 8 | Native NN-descent k32 | 48.944 | 0.991370 |
| YFCC-10M | 8 | Rebuild | 23.001 | 0.988518 |
| YFCC-10M | 128 | Fastener binary | 2.484 | 0.961816 |
| YFCC-10M | 128 | Fastener ternary | 2.695 | 0.962917 |
| YFCC-10M | 128 | Native NN-descent k32 | 48.831 | 0.989372 |
| YFCC-10M | 128 | Rebuild | 23.137 | 0.988062 |
