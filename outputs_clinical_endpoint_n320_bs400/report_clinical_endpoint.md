# 临床可读汇报 — endpoint 二分类

对照提示：若以此前 **`outputs_deep_full_gpu_sprint3`** 中用「≤3年时间窗二元代理标签」训练的树/MLP 为基线，本轮已将监督标签统一为 **`endpoint` 本体二分类**（仍保留 Cox 与时间依赖 AUC 作为辅助解读）。在同一外层划分与可比 seed 设置下，应重点对比 **`metrics_endpoint_binary_test.csv`**（AUC/BalancedAcc/F1 @0.5、Brier/ECE）与 **`clinical_thresholds_test_performance.csv`**。

## 阈值推荐（训练折 OOF 上定义；外测复述）
- 主方案：`max_balanced_accuracy`。
- 备选：`max_youden_j`、`conservative_high_specificity`、`aggressive_high_sensitivity`。

### 训练侧 OOF 上各规则对应的阈值与指标摘要
```
 threshold  balanced_accuracy       f1      mcc  sensitivity  specificity      ppv      npv   lr_pos   lr_neg  youden_j                          rule                     rule_name
  0.258872           0.747772 0.543210 0.439130     0.647059     0.848485 0.468085 0.921053 4.270588 0.415966  0.495544         max_balanced_accuracy         max_balanced_accuracy
  0.258872           0.747772 0.543210 0.439130     0.647059     0.848485 0.468085 0.921053 4.270588 0.415966  0.495544                  max_youden_j                  max_youden_j
  0.067562           0.694474 0.426230 0.294761     0.764706     0.624242 0.295455 0.927928 2.035104 0.376927  0.388948 conservative_high_specificity conservative_high_specificity
  0.130484           0.728699 0.484848 0.367085     0.705882     0.751515 0.369231 0.925373 2.840746 0.391366  0.457398   aggressive_high_sensitivity   aggressive_high_sensitivity
```

### 固定外测（复述各阈值对应的 Sens/Spec/PPV/NPV/LR+/LR-/F1/MCC）
```
{thr_tab}
```

## 概率层面的 endpoint 判别（阈值=0.5 及校准）
```
                   model      auc  balanced_accuracy_050   f1_050  mcc_050    brier      ece  calibration_slope_logit  calibration_intercept_logit
    model4_tree_endpoint 0.691558               0.663961 0.470588 0.425423 0.119401 0.113435                 0.423730                    -0.685663
     model5_mlp_endpoint 0.806818               0.684253 0.424242 0.290682 0.172921 0.173543                 0.390884                    -1.201370
model6_ensemble_endpoint 0.785714               0.701299 0.480000 0.366825 0.131746 0.149403                 0.534991                    -0.827563
```

## 图示
- `figures/calibration_curve_endpoint_ensemble.png`
- `figures/decision_curve_endpoint_ensemble.png`
（DCA 与同图所用概率为固定外测上的模型融合概率。）
